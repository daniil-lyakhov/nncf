# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch
from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils import LOGGER
from ultralytics.utils import RANK
from ultralytics.utils import SETTINGS
from ultralytics.utils import __version__
from ultralytics.utils import checks
from ultralytics.utils import yaml_load
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.torch_utils import strip_optimizer

import nncf
from nncf import IgnoredScope
from nncf.torch import load_from_aux
from nncf.torch.model_creation import is_wrapped_model

CHECKPOINT_PATH = "yolov8n.pt"
# CHECKPOINT_PATH = "path/to/saved/ckpt"
# CHECKPOINT_PATH = "path/to/saved/ckpt"
# CHECKPOINT_PATH = "/home/dlyakhov/Projects/ultralytics/runs/detect/train92/weights/best.pt"


class QuantizationTrainer(DetectionTrainer):
    def __init__(
        self,
        ignored_scope: Optional[IgnoredScope] = None,
        cfg=DEFAULT_CFG,
        overrides=None,
        _callbacks=None,
        nncf_config=None,
        quantization_state_dict=None,
    ):
        super().__init__(cfg, overrides, _callbacks)
        self._nncf_ignored_scope = IgnoredScope(patterns=[".*/Detect.*"]) if ignored_scope is None else ignored_scope
        self.nncf_dataloader = None
        self._nncf_config = nncf_config
        self._quantization_state_dict = quantization_state_dict

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if self._nncf_config is None or self._quantization_state_dict is None:
            if weights:
                model.load(weights)
            return model
        # Quantized model was saved outside the training loop.
        # Needs to be recovered here from the checkpoint
        self.original_model = deepcopy(model)

        if not (isinstance(weights, torch.nn.Module) and is_wrapped_model(weights)):
            # Hack to get inputs
            model_ref = self.model
            self.model = model
            example_input = next(iter(self.get_nncf_dataset().get_inference_data()))
            self.model = model_ref
        else:
            args, kwargs = weights.nncf.input_infos.get_forward_inputs()
            example_input = args[0]
        model = load_from_aux(model.to(self.device), self._nncf_config, example_input)
        # model.load(weights)
        model.load_state_dict(self._quantization_state_dict)
        self._quantization_state_dict = None
        self._nncf_config = None
        return model

    def setup_model(self):
        ckpt = super().setup_model()

        if not is_wrapped_model(self.model):
            # Make copy of model to support `DetectionTrainer` save/load logic
            self.original_model = deepcopy(self.model)
            if ckpt is not None and ckpt.get("NNCF_AUX_CONFIG"):
                self.resume_model_for_qat(ckpt)
            else:
                self.prepare_model_for_qat()
        return ckpt

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        # Disable EMA for QAT. Using EMA may reduce the accuracy of the model during training.
        if self.ema:
            self.ema.enabled = False

    def get_nncf_dataset(self):
        if self.nncf_dataloader is None:

            def transform_fn(x):
                x = self.preprocess_batch(x)
                return x["img"]

            train_loader = self.get_dataloader(self.testset, batch_size=1, rank=RANK, mode="train")
            self.nncf_dataloader = nncf.Dataset(train_loader, transform_fn)
        return self.nncf_dataloader

    def prepare_model_for_qat(self):
        calibration_dataset = self.get_nncf_dataset()
        self.model = nncf.quantize(
            self.model.to(self.device), calibration_dataset, ignored_scope=self._nncf_ignored_scope
        )

    def resume_model_for_qat(self, ckpt):
        example_input = next(iter(self.get_nncf_dataset().get_inference_data()))
        self.model = load_from_aux(self.model.to(self.device), ckpt["NNCF_AUX_CONFIG"], example_input)
        self.model.load_state_dict(ckpt["model_state_dict"])

    def save_qat_model(self):
        import pandas as pd  # scope for faster startup

        metrics = {**self.metrics, **{"fitness": self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()}

        aux_config = self.model.nncf.get_aux_config()

        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.original_model)).half(),
            "model_state_dict": de_parallel(self.model).state_dict(),
            "NNCF_AUX_CONFIG": aux_config,
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),  # save as dict
            "train_metrics": metrics,
            "train_results": results,
            "date": datetime.now().isoformat(),
            "version": __version__,
        }

        # Save last and best
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f"epoch{self.epoch}.pt")
        del ckpt

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.model = f
                    self.setup_model()
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=self.model)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def save_model(self):
        if is_wrapped_model(self.model):
            self.save_qat_model()
        else:
            super().save_model()


class YOLOINT8(YOLO):
    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        super().__init__(model, task, verbose)
        self._nncf_config = None
        self._original_model = None

    def quantize(self, **kwargs):
        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {"data": DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task]}  # method defaults
        args = {**overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        self.trainer = QuantizationTrainer(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume"):  # manually set model only if not resuming
            # Current model has no affect on setup model. self.trainer.get_model should be used here with
            # weights == self.model
            self._original_model = deepcopy(self.model)
            self.trainer.setup_model()
            self.model = self.trainer.model
            self._nncf_config = self.model.nncf.get_aux_config()

    def train(
        self,
        trainer=None,
        **kwargs,
    ):
        qat = False
        if "NNCF_AUX_CONFIG" in self.ckpt:
            nncf_config = self.ckpt["NNCF_AUX_CONFIG"]
            quantization_state_dict = self.ckpt["quantization_state_dict"]
            qat = True

        self._check_is_pytorch_model()
        if hasattr(self.session, "model") and self.session.model.id:  # Ultralytics HUB session with loaded model
            if any(kwargs):
                LOGGER.warning("WARNING ⚠️ using HUB training arguments, ignoring local training arguments.")
            kwargs = self.session.train_args  # overwrite kwargs

        checks.check_pip_update_available()

        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {"data": DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task]}  # method defaults
        args = {**overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        if qat:
            self.trainer = QuantizationTrainer(
                overrides=args,
                _callbacks=self.callbacks,
                nncf_config=nncf_config,
                quantization_state_dict=quantization_state_dict,
            )
        else:
            self.trainer = QuantizationTrainer(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

            if SETTINGS["hub"] is True and not self.session:
                # Create a model in HUB
                try:
                    self.session = self._get_hub_session(self.model_name)
                    if self.session:
                        self.session.create_model(args)
                        # Check model was created
                        if not getattr(self.session.model, "id", None):
                            self.session = None
                except (PermissionError, ModuleNotFoundError):
                    # Ignore PermissionError and ModuleNotFoundError which indicates hub-sdk not installed
                    pass

        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
            if qat:
                self.trainer.model = ckpt
                self.trainer.setup_model()
                self.model = self.trainer.model
        return self.metrics

    def save(self, filename: Union[str, Path] = "saved_model.pt", use_dill=True):
        if self._nncf_config is not None:
            self.ckpt["NNCF_AUX_CONFIG"] = self._nncf_config
            self.ckpt["quantization_state_dict"] = self.model.state_dict()
            self.ckpt["model"] = self._original_model
        ret_val = super().save(filename, use_dill)
        if self._nncf_config is not None:
            self.ckpt["model"] = self.model
            del self.ckpt["NNCF_AUX_CONFIG"]
            del self.ckpt["quantization_state_dict"]
        return ret_val

    @property
    def task_map(self):
        map = super().task_map
        map["detect"]["trainer"] = QuantizationTrainer
        return map


def main():
    # args = dict(model=CHECKPOINT_PATH, data="coco8.yaml", epochs=0, mode="train", verbose=False)
    # model_args = dict(model=CHECKPOINT_PATH, data="coco8.yaml")
    # val_args = dict(model=CHECKPOINT_PATH, data="coco8.yaml")
    model = YOLO()
    original_val = model.val(data="coco8.yaml")
    model = YOLOINT8()
    model.quantize(data="coco8.yaml")
    ckpt_path = "yolo8_int8.pt"
    model.save(ckpt_path)
    quantized_init_val = model.val(data="coco8.yaml")
    model = YOLOINT8(ckpt_path)
    model.train(data="coco8.yaml", epochs=10)
    quantized_tuned_val = model.val(data="coco8.yaml")
    print("Metric: mAP50")
    print(f"Baseline: {original_val.box.map50}")
    print(f"INT8 init: {quantized_init_val.box.map50}")
    print(f"INT8 tuned: {quantized_tuned_val.box.map50}")


if __name__ == "__main__":
    main()
