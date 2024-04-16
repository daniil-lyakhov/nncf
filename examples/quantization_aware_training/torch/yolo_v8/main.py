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

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import LOGGER
from ultralytics.utils import RANK
from ultralytics.utils import __version__
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.torch_utils import strip_optimizer

import nncf
from nncf.torch import load_from_aux
from nncf.torch.model_creation import is_wrapped_model

# CHECKPOINT_PATH = "yolov8n.pt"
# CHECKPOINT_PATH = "path/to/saved/ckpt"
# CHECKPOINT_PATH = "path/to/saved/ckpt"
CHECKPOINT_PATH = "/home/dlyakhov/Projects/ultralytics/runs/detect/train92/weights/best.pt"


class MyTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.nncf_dataloader = None

    def setup_model(self):
        ckpt = super().setup_model()

        if not is_wrapped_model(self.model):
            # Make copy of model to support `DetectionTrainer` save/load logic
            self.original_model = deepcopy(self.model)
            if ckpt.get("NNCF_AUX_CONFIG"):
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
                return x["img"], None

            train_loader = self.get_dataloader(self.trainset, batch_size=1, rank=RANK, mode="train")
            self.nncf_dataloader = nncf.Dataset(train_loader, transform_fn)
        return self.nncf_dataloader

    def prepare_model_for_qat(self):
        calibration_dataset = self.get_nncf_dataset()
        # TODO figure out why inputs are being quantized. Do conv.pre_hook instead of input.post_hook
        ignored_scope = nncf.IgnoredScope(
            names=["DetectionModel/Sequential[model]/Conv[0]/NNCFConv2d[conv]/conv2d_0"], patterns=[".*/Detect.*"]
        )
        self.model = nncf.quantize(self.model.to(self.device), calibration_dataset, ignored_scope=ignored_scope)

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


def main():
    args = dict(model=CHECKPOINT_PATH, data="coco8.yaml", epochs=3, mode="train", verbose=False)
    nncf_trainer = MyTrainer(overrides=args)
    nncf_trainer.train()


if __name__ == "__main__":
    main()
