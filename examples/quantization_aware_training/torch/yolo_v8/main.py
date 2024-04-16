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
from typing import Any, Dict, Tuple

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import LOGGER
from ultralytics.utils import RANK
from ultralytics.utils import __version__
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.torch_utils import strip_optimizer

from nncf import NNCFConfig
from nncf.torch import create_compressed_model
from nncf.torch import register_default_init_args
from nncf.torch.dynamic_graph.io_handling import nncf_model_input
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.model_creation import is_wrapped_model


# 1 integration issue:
# MyInitializingDataLoader must support deep copy because DetectionTrainer does a deep copy
# of the model and MyInitializingDataLoader during training setup. The input data_loader
# of ultralytics.data.build.InfiniteDataLoader type does not support deep copy and
# can not be used directly into MyInitializingDataLoader. The workaround for this limitation is
# to create a deepcopable dataset from the data_loader.
class MyInitializingDataLoader(PTInitializingDataLoader):
    def __init__(self, data_loader, preprocess_batch_fn, num_samples=300):
        super().__init__(data_loader)
        self._batch_size = self._data_loader.batch_size
        # Using list of images instead of 'ultralytics.data.build.InfiniteDataLoader' to support deepcopy.
        self._data_loader = []
        num_samples = num_samples / self._batch_size
        for count, data_item in enumerate(data_loader):
            if count > num_samples:
                break
            batch = preprocess_batch_fn(data_item)
            self._data_loader.append((batch["img"], None))

    @property
    def batch_size(self):
        return self._batch_size

    def get_inputs(self, dataloader_output: Any) -> Tuple[Tuple, Dict]:
        # your implementation - `dataloader_output` is what is returned by your dataloader,
        # and you have to turn it into a (args, kwargs) tuple that is required by your model
        # in this function, for instance, if your dataloader returns dictionaries where
        # the input image is under key `"img"`, and your YOLOv8 model accepts the input
        # images as 0-th `forward` positional arg, you would do:
        return (dataloader_output[0],), {}

    def get_target(self, dataloader_output: Any) -> Any:
        # and in this function you should extract the "ground truth" value from your
        # dataloader, so, for instance, if your dataloader output is a dictionary where
        # ground truth images are under a "gt" key, then here you would write:
        return dataloader_output[1]


class MyTrainer(DetectionTrainer):
    def __init__(self, nncf_config_dict, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.nncf_config = NNCFConfig.from_dict(nncf_config_dict)
        self.nncf_dataloader = None

    def setup_model(self):
        ckpt = super().setup_model()

        if not is_wrapped_model(self.model):
            # Make copy of model to support `DetectionTrainer` save/load logic
            self.original_model = deepcopy(self.model)
            if ckpt.get("model_compression_state"):
                self.resume_model_for_qat(ckpt)
            else:
                self.prepare_model_for_qat()
        return ckpt

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        # Disable EMA for QAT. Using EMA may reduce the accuracy of the model during training.
        if self.ema:
            self.ema.enabled = False

    def get_nncf_dataloader(self):
        if self.nncf_dataloader is None:
            num_samples = self.nncf_config["compression"]["initializer"]["range"]["num_init_samples"]
            train_loader = self.get_dataloader(self.trainset, batch_size=1, rank=RANK, mode="train")
            self.nncf_dataloader = MyInitializingDataLoader(train_loader, self.preprocess_batch, num_samples)
        return self.nncf_dataloader

    def create_wrap_inputs_fn(self):
        # 2 integration issue:
        # NNCF requires the same structure of inputs in the forward function during model training
        # for correct model tracing, but the DetectionModel forward function support image tensor
        # or dict as input:
        # def forward(self, x, *args, **kwargs):
        #     if isinstance(x, dict):  # for cases of training and validating while training.
        #         return self.loss(x, *args, **kwargs)
        #     return self.predict(x, *args, **kwargs)
        # In this case, wrap_inputs_fn should be implemented to specify the "original" model input
        def wrap_inputs_fn(args, kwargs):
            if isinstance(args[0], dict):
                return args, kwargs
            args = (nncf_model_input(args[0]),) + args[1:]
            return args, kwargs

        return wrap_inputs_fn

    def prepare_model_for_qat(self):
        nncf_dataloader = self.get_nncf_dataloader()
        self.nncf_config = register_default_init_args(self.nncf_config, nncf_dataloader)

        self.model = self.model.to(self.device)
        _, self.model = create_compressed_model(
            self.model, self.nncf_config, wrap_inputs_fn=self.create_wrap_inputs_fn()
        )

    def resume_model_for_qat(self, ckpt):
        _, self.model = create_compressed_model(
            self.model,
            self.nncf_config,
            compression_state=ckpt["model_compression_state"],
            wrap_inputs_fn=self.create_wrap_inputs_fn(),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])

    def save_qat_model(self):
        import pandas as pd  # scope for faster startup

        metrics = {**self.metrics, **{"fitness": self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()}

        compression_controller = self.model.nncf.compression_controller
        model_compression_state = {}
        if compression_controller is not None:
            model_compression_state = compression_controller.get_compression_state()

        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.original_model)).half(),
            "model_state_dict": de_parallel(self.model).state_dict(),
            "model_compression_state": model_compression_state,
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
    args = dict(model="yolov8n.pt", data="coco8.yaml", epochs=3, mode="train", verbose=False)
    nncf_config_dict = {
        "input_info": {"sample_size": [1, 3, 640, 640]},
        "log_dir": "yolov8_output",  # The log directory for NNCF-specific logging outputs.
        "compression": {
            "algorithm": "quantization",
            "ignored_scopes": ["{re}/Detect"],  # ignored the post-processing
            "initializer": {"range": {"num_init_samples": 300}},
        },
    }
    nncf_trainer = MyTrainer(nncf_config_dict, overrides=args)
    nncf_trainer.train()


if __name__ == "__main__":
    main()
