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
from pathlib import Path
from typing import Dict, Tuple

import cv2
import openvino as ov
import openvino.torch  # noqa
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator as Validator
from ultralytics.models.yolo import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel

import nncf
from nncf.torch import disable_patching

MODEL_NAME = "yolo11n"

ROOT = Path(__file__).parent.resolve()


def quantize(model: ov.Model, data_loader: torch.utils.data.DataLoader, validator: DetectionValidator) -> ov.Model:
    def transform_fn(data_item: Dict):
        """
        Quantization transform function. Extracts and preprocess input data from dataloader
        item for quantization.
        Parameters:
        data_item: Dict with data item produced by DataLoader during iteration
        Returns:
            input_tensor: Input data for quantization
        """
        input_tensor = validator.preprocess(data_item)["img"]
        return input_tensor

    quantization_dataset = nncf.Dataset(data_loader, transform_fn)
    quantized_model = nncf.quantize(
        model,
        quantization_dataset,
        subset_size=len(data_loader),
        preset=nncf.QuantizationPreset.MIXED,
        ignored_scope=nncf.IgnoredScope(
            types=["mul", "sub", "sigmoid", "__getitem__"],
            subgraphs=[
                nncf.Subgraph(
                    inputs=["cat_13", "cat_14", "cat_15"],
                    outputs=["output"],
                )
            ],
        ),
    )
    return quantized_model


def _prepare_validation(model: YOLO, data: str) -> Tuple[Validator, torch.utils.data.DataLoader]:
    custom = {"rect": False, "batch": 1}  # method defaults
    args = {**model.overrides, **custom, "mode": "val"}  # highest priority args on the right

    validator = model._smart_load("validator")(args=args, _callbacks=model.callbacks)
    stride = 32  # default stride
    validator.stride = stride  # used in get_dataloader() for padding
    validator.data = check_det_dataset(data)
    validator.init_metrics(de_parallel(model))

    data_loader = validator.get_dataloader(validator.data.get(validator.args.split), validator.args.batch)

    return validator, data_loader


# ultralytics==8.3.27
def main():
    model = YOLO(ROOT / f"{MODEL_NAME}.pt")

    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = "coco128.yaml"
    validator, data_loader = _prepare_validation(model, "coco128.yaml")
    import numpy as np

    np_dummy_tensor = np.ones((360, 640, 3))
    breakpoint()
    model(np_dummy_tensor)

    pt_model = model.model
    # Run mode one time to initialize all
    # internal variables
    # pt_model(dummy_tensor)

    dummy_tensor = torch.ones((1, 3, 384, 640))
    dummy_tensor = torch.ones((1, 3, 640, 640))
    with torch.no_grad():
        with disable_patching():
            pass
            # pt_model = torch.export.export(pt_model, args=(dummy_tensor,), strict=True).module()
            # pt_model = torch.compile(pt_model)
            pt_model = torch.export.export(pt_model, args=(dummy_tensor,), strict=False).module()
            # pt_model = torch.compile(pt_model, backend="openvino")

    pt_model = quantize(pt_model, data_loader, validator)
    pt_model = torch.export.export(pt_model, args=(torch.ones((1, 3, 384, 640)),), strict=False).module()
    pt_model = torch.compile(pt_model, backend="openvino")
    # exit()
    # model.predict(source="https://youtu.be/4aWufTZDLMU?si=Vd24h_w39XJb1PgX", save=True)
    # model.predict(source="tcp://127.0.0.1:23000", save=True, name="ov_stream")

    # Open the video file
    video_path = (
        "/home/dlyakhov/Projects/nncf/examples/post_training_quantization/openvino/yolov8/Camera_road_in_Thailand.mp4"
    )
    save_path = "out"
    cap = cv2.VideoCapture(video_path)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    suffix, fourcc = (".avi", "MJPG")
    vid_writer = cv2.VideoWriter(
        filename=str(Path(save_path).with_suffix(suffix)),
        fourcc=cv2.VideoWriter_fourcc(*fourcc),
        fps=30,  # integer required, floats produce error in MP4 codec
        frameSize=(int(width), int(height)),  # (width, height)
    )

    idx = 0
    sec = 10
    fps = 30
    while cap.isOpened():
        # Read a frame from the video
        idx += 1
        if idx > fps * sec:
            break

        if idx % fps == 0:
            print(f"{idx // fps}/{sec}")

        success, frame = cap.read()

        if not success:
            break
        # Run YOLO inference on the frame

        # frame = torch.tensor(frame).transpose(0, -1).unsqueeze(0)
        pre_frame = model.predictor.preprocess([frame])
        results = pt_model(pre_frame)
        results = model.predictor.postprocess(results, pre_frame, [frame])

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        vid_writer.write(annotated_frame)

    # Release the video capture object and close the display window
    cap.release()
    vid_writer.release()


if __name__ == "__main__":
    main()
