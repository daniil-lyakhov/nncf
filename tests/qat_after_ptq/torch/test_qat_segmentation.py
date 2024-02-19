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

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.utils.data
import torch.utils.data.distributed
from helpers import broadcast_initialized_parameters
from helpers import get_advanced_ptq_parameters
from helpers import get_mocked_compression_ctrl
from helpers import get_num_samples
from helpers import get_quantization_preset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import nncf
from examples.common.sample_config import SampleConfig
from examples.common.sample_config import create_sample_config
from examples.torch.common.example_logger import logger
from examples.torch.common.execution import get_execution_mode
from examples.torch.common.execution import prepare_model_for_execution
from examples.torch.common.execution import start_worker
from examples.torch.common.model_loader import load_model
from examples.torch.common.optimizer import make_optimizer
from examples.torch.common.utils import configure_device
from examples.torch.common.utils import configure_logging
from examples.torch.common.utils import is_pretrained_model_requested
from examples.torch.semantic_segmentation.main import get_arguments_parser
from examples.torch.semantic_segmentation.main import get_criterion
from examples.torch.semantic_segmentation.main import get_dataset
from examples.torch.semantic_segmentation.main import get_joint_transforms
from examples.torch.semantic_segmentation.main import get_params_to_optimize
from examples.torch.semantic_segmentation.main import load_dataset
from examples.torch.semantic_segmentation.main import test as sample_validate
from examples.torch.semantic_segmentation.metric import IoU
from examples.torch.semantic_segmentation.test import Test
from examples.torch.semantic_segmentation.train import Train
from nncf import NNCFConfig
from nncf.torch.utils import is_main_process
from tests.shared.paths import PROJECT_ROOT

CONFIGS = list((PROJECT_ROOT / Path("examples/torch/semantic_segmentation/configs")).glob("*"))


@pytest.fixture(name="quantization_config_path", params=CONFIGS, ids=[conf.stem for conf in CONFIGS])
def fixture_quantization_config(request):
    return request.param


def get_sample_config(quantization_config_path: Path, data_dir: Path, weights_dir: Path) -> SampleConfig:
    parser = get_arguments_parser()
    weights_path = weights_dir / (quantization_config_path.stem.split("_int8")[0] + ".pth")
    meta = None
    datasets_meta = [{"name": "mapillary", "dir_name": "mapillary_vistas"}, {"name": "camvid", "dir_name": "camvid"}]
    for datset_meta in datasets_meta:
        if datset_meta["name"] in quantization_config_path.stem:
            meta = datset_meta
            break
    else:
        raise RuntimeError(f"Dataset for the config {str(quantization_config_path)} is unknown.")

    data_dir = data_dir / meta["dir_name"]
    args = parser.parse_args(
        [
            "-c",
            str(quantization_config_path),
            "--data",
            str(data_dir),
            "--dataset",
            meta["name"],
            "--weights",
            str(weights_path),
        ]
    )
    sample_config = create_sample_config(args, parser)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    sample_config.device = device
    sample_config.execution_mode = get_execution_mode(sample_config)
    return sample_config


@dataclass
class DatasetSet:
    train_data_loader: torch.utils.data.DataLoader
    val_data_loader: torch.utils.data.DataLoader
    class_weights: object
    calibration_dataset: nncf.Dataset


def get_datasets(dataset, config: SampleConfig) -> DatasetSet:
    loaders, w_class = load_dataset(dataset, config)
    train_loader, val_loader, _ = loaders
    transforms_val = get_joint_transforms(is_train=False, config=config)
    # Get selected dataset
    val_dataset = dataset(config.dataset_dir, image_set="val", transforms=transforms_val)

    def transform_fn(data_item):
        return data_item[0].to(config.device)

    val_data_loader_batch_one = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    calibration_dataset = nncf.Dataset(val_data_loader_batch_one, transform_fn)
    return DatasetSet(
        train_data_loader=train_loader,
        val_data_loader=val_loader,
        class_weights=w_class,
        calibration_dataset=calibration_dataset,
    )


def test_compression_training(quantization_config_path: Path, data_dir: Path, weights_dir: Path, mocker):
    nncf_config = NNCFConfig.from_json(quantization_config_path)
    if (
        "compression" not in nncf_config
        or isinstance(nncf_config["compression"], list)
        or nncf_config["compression"]["algorithm"] != "quantization"
    ):
        pytest.skip("Config without compression")

    config = get_sample_config(quantization_config_path, data_dir, weights_dir)
    if "accuracy_aware_training" in config:
        pytest.skip("Accuracy Aware training is not supported yet for QAT with PTQ.")

    start_worker(main_worker, config)


def main_worker(current_gpu: int, config: SampleConfig):
    configure_device(current_gpu, config)
    if is_main_process():
        configure_logging(logger, config)

    # create model
    logger.info(f"\nCreating model from config: {config.config}")

    dataset = get_dataset(config.dataset)
    color_encoding = dataset.color_encoding
    num_classes = len(color_encoding)

    pretrained = is_pretrained_model_requested(config)
    model = load_model(
        config.model,
        pretrained=pretrained,
        num_classes=num_classes,
        model_params=config.get("model_params", {}),
        weights_path=config.get("weights"),
    )
    model.to(config.device)

    datasets = get_datasets(dataset, config)
    criterion = get_criterion(datasets.class_weights, config)

    logger.info("Original model validation:")
    original_metric = sample_validate(model, datasets.val_data_loader, criterion, color_encoding, config)

    logger.info("Apply quantization to the model:")
    config_quantization_params = config["compression"]

    preset = get_quantization_preset(config_quantization_params)
    advanced_parameters = get_advanced_ptq_parameters(config_quantization_params)
    subset_size = get_num_samples(config_quantization_params)

    quantized_model = nncf.quantize(
        model,
        datasets.calibration_dataset,
        preset=preset,
        advanced_parameters=advanced_parameters,
        subset_size=subset_size,
    )
    model, model_without_dp = prepare_model_for_execution(model, config)
    if config.distributed:
        broadcast_initialized_parameters(model)

    acc_drop = train(
        quantized_model,
        model_without_dp,
        config,
        criterion,
        datasets,
        original_metric,
        color_encoding,
        get_mocked_compression_ctrl(),
    )
    assert accuracy_drop_is_acceptable(acc_drop)


def accuracy_drop_is_acceptable(acc_drop: float) -> bool:
    """
    Returns True in case acc_drop is less than 1 precent.
    """
    return acc_drop < 0.01


def train(
    model: torch.nn.Module,
    model_without_dp: torch.nn.Module,
    config: SampleConfig,
    criterion: torch.nn.Module,
    datasets: DatasetSet,
    original_metric: float,
    color_encoding: object,
    compression_ctrl,
) -> float:
    """
    :return: Accuracy drop between original accuracy and trained quantized model accuracy.
    """
    logger.info("\nTraining...\n")

    optim_config = config.get("optimizer", {})
    optim_params = optim_config.get("optimizer_params", {})
    lr = optim_params.get("lr", 1e-4)

    params_to_optimize = get_params_to_optimize(model_without_dp, lr * 10, config)
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)

    # Evaluation metric

    ignore_index = None
    ignore_unlabeled = config.get("ignore_unlabeled", True)
    if ignore_unlabeled and ("unlabeled" in color_encoding):
        ignore_index = list(color_encoding).index("unlabeled")

    metric = IoU(len(color_encoding), ignore_index=ignore_index)

    best_miou = -1

    # Start Training
    train_obj = Train(
        model, datasets.train_data_loader, optimizer, criterion, compression_ctrl, metric, config.device, config.model
    )
    val_obj = Test(model, datasets.val_data_loader, criterion, metric, config.device, config.model)

    for epoch in range(config.start_epoch, config.epochs):
        if config.distributed:
            datasets.train_data_loader.sampler.set_epoch(epoch)

        logger.info(">>>> [Epoch: {0:d}] Validation".format(epoch))
        loss, (iou, current_miou) = val_obj.run_epoch(config.print_step)
        # best_metric = max(current_miou, best_metric)
        acc_drop = original_metric - current_miou
        best_miou = max(current_miou, best_miou)
        logger.info(f"Metric: {current_miou}, FP32 diff: {acc_drop}")
        if accuracy_drop_is_acceptable(acc_drop):
            logger.info(f"Accuracy is within 1 percent drop," f" pipeline is making early exit on epoch {epoch - 1}")
            logger.info(
                f"Epochs in config: {config.epochs}, epochs trained: {epoch}, epochs saved: {config.epochs - epoch}"
            )
            return acc_drop
        if epoch == config.epochs:
            logger.info("Training pipeline is finished, accuracy was not recovered.")
            return acc_drop

        logger.info(">>>> [Epoch: {0:d}] Training".format(epoch))
        epoch_loss, (iou, miou) = train_obj.run_epoch(config.print_step)

        logger.info(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, epoch_loss, miou))

        lr_scheduler.step(epoch if not isinstance(lr_scheduler, ReduceLROnPlateau) else best_miou)