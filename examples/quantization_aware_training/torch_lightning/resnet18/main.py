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

import os
from copy import deepcopy
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.quantization import QuantizationAwareTraining
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import StepLR
from torchmetrics.functional import accuracy

import nncf

seed_everything(7)


class NNCFQuantizationAwareTraining(QuantizationAwareTraining):
    """Callback for NNCF compression.

    Assumes that the pl module contains a 'model' attribute, which is
    the PyTorch module that must be compressed.

    Args:
        config (dict): NNCF Configuration
        export_dir (Str): Path where the export `onnx` and the OpenVINO `xml` and `bin` IR are saved.
                          If None model will not be exported.
    """

    def __init__(self, initialize_quantization=True) -> None:
        self._initialize_quantization = initialize_quantization

        def transform_fn(input_):
            return torch.unsqueeze(input_[0][0], 0)

        self._transform_fn = transform_fn
        self._state = None
        self._input_shape = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        """Call when fit or test begins.

        Takes the pytorch model and wraps it using the compression controller
        so that it is ready for nncf fine-tuning.
        """
        if not self._initialize_quantization or hasattr(pl_module.model, "nncf"):
            return

        import nncf

        nncf_dataset = nncf.Dataset(trainer.datamodule.val_dataloader(), self._transform_fn)
        pl_module.model = nncf.quantize(pl_module.model, calibration_dataset=nncf_dataset, subset_size=1)

        self._state = pl_module.model.nncf.transformations_config()
        example_input = self._transform_fn(next(iter(trainer.datamodule.val_dataloader())))
        self._input_shape = tuple(example_input.shape)

    def state_dict(self) -> Dict[str, Any]:
        return {"state": self._state, "input_shape": self._input_shape}

    def _load_before_model(self, model: pl.LightningModule, state_dict: Dict[str, Any]) -> None:
        """Special hook that gets called by the CheckpointConnector *before* the model gets loaded."""
        load_transformation_from_state_dict(model, state_dict)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return


def load_transformation_from_state_dict(model: pl.LightningModule, state_dict: Dict[str, Any]) -> None:
    input_shape = state_dict["input_shape"]
    transformation_config = state_dict["state"]

    transformed_model = nncf.torch.from_config(deepcopy(model.model), transformation_config, torch.ones(input_shape))
    model.model = transformed_model


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

"""### CIFAR10 Data Module

Import the existing data module from `bolts` and modify the train and test transforms.
"""

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

"""### Resnet
Modify the pre-existing Resnet architecture from TorchVision. The pre-existing architecture is based on ImageNet
images (224x224) as input. So we need to modify it for CIFAR10 images (32x32).
"""


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


"""### Lightning Module
Check out the [`configure_optimizers`](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers)
method to use custom Learning Rate schedulers. The OneCycleLR with SGD will get you to around 92-93% accuracy
in 20-30 epochs and 93-94% accuracy in 40-50 epochs. Feel free to experiment with different
LR schedules from https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
"""


class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler_dict = {
            "scheduler": StepLR(
                optimizer,
                0.1,
            ),
            "interval": "epoch",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if hasattr(self.model, "nncf"):
            return
        load_transformation_from_state_dict(self, checkpoint["callbacks"][NNCFQuantizationAwareTraining.__name__])


def get_model_and_trainer(max_epochs=1, initialize_quantization=True):
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        auto_lr_find=True,
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[
            NNCFQuantizationAwareTraining(initialize_quantization=initialize_quantization),
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
        ],
    )
    model = LitResnet()
    return model, trainer


print("[STAGE 1] Quantize and train the model")
model, trainer = get_model_and_trainer()
trainer.fit(model, cifar10_dm)
trainer.test(model, datamodule=cifar10_dm)

ckpt_path = "ckpt.pt"
trainer.save_checkpoint(ckpt_path)

print("[STAGE 2] Model and trainer are deleted")
del model
del trainer

print("Recover model with quantization from the Trainer checkpoint by the NNCF Trainer callback and train the model")
model, trainer = get_model_and_trainer(2, initialize_quantization=False)
trainer.fit(model, datamodule=cifar10_dm, ckpt_path=ckpt_path)

print("[STAGE 3] Model and trainer are deleted")
del model
del trainer

print(
    "[STAGE 4] Recover model with quantization from the Trainer checkpoint"
    "by the NNCF Trainer callback and test the model"
)
model, trainer = get_model_and_trainer(2, initialize_quantization=False)
trainer.test(model, datamodule=cifar10_dm)

print("[STAGE 5] Model and trainer are deleted")
del model
del trainer

print(
    "[STAGE 6] Recover model with quantization from the Trainer checkpoint by the `on_load_checkpoint` pl.Module method"
    " and test the model with a new Trainer"
)
model, trainer = get_model_and_trainer(initialize_quantization=False)
model.load_from_checkpoint(ckpt_path)
trainer.test(model, datamodule=cifar10_dm)
