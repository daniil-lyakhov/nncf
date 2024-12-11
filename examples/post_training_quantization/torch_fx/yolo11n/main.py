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
from typing import Iterator

import cv2
import numpy as np
import openvino as ov
import openvino.torch  # noqa
import torch
from ultralytics.models.yolo import YOLO

import nncf
from nncf.torch import disable_patching

MODEL_NAME = "yolo11n"

ROOT = Path(__file__).parent.resolve()


class CV2VideoIter:
    def __init__(self, cap) -> None:
        self._cap = cap

    def __iter__(self):
        return self

    def __next__(self):
        success, frame = self._cap.read()
        if not success:
            raise StopIteration()
        return frame

    def __len__(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))


class CV2VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, cap) -> None:
        super().__init__()
        self._iter = CV2VideoIter(cap)

    def __iter__(self) -> Iterator:
        return self._iter

    def __len__(self):
        return len(self._iter)


def quantize(model: ov.Model, data_loader: CV2VideoDataset, transform_fn) -> ov.Model:

    quantization_dataset = nncf.Dataset(data_loader, transform_fn)
    quantized_model = nncf.quantize(
        model,
        quantization_dataset,
        subset_size=300,
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


# ultralytics==8.3.27
def main():
    model = YOLO(ROOT / f"{MODEL_NAME}.pt")

    # Open the video file
    video_path = (
        # "/home/dlyakhov/Projects/nncf/examples/
        # post_training_quantization/torch_fx/yolo11n/Camera_road_in_Thailand.mp4"
        "/home/dlyakhov/Projects/nncf/examples/post_training_quantization/torch_fx/yolo11n/animals.mp4"
    )

    save_path = "out_int8"
    save_path = "out"
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Setup pre-processing
    np_dummy_tensor = np.ones((height, width, 3))
    model.predict(np_dummy_tensor, imgsz=((height, width)), device="cpu")
    # model.predict(frame)

    pt_model = model.model.to(torch.device("cpu"))
    # Run mode one time to initialize all
    # internal variables
    # pt_model(dummy_tensor)

    def transform_fn(frame):
        input_tensor = model.predictor.preprocess([frame])
        return input_tensor

    with torch.no_grad():
        with disable_patching():
            pass
            # pt_model = torch.export.export(pt_model, args=(dummy_tensor,), strict=True).module()
            # pt_model = torch.compile(pt_model)
            pt_model = torch.export.export(pt_model, args=(transform_fn(np_dummy_tensor),), strict=False).module()
            # pt_model = torch.compile(pt_model, backend="openvino")

    # pt_model = quantize(pt_model, CV2VideoDataset(cap), transform_fn)
    # pt_model = torch.export.export(pt_model, args=(torch.ones((1, 3, 384, 640)),), strict=False).module()
    pt_model = torch.compile(pt_model, backend="openvino")
    # exit()
    # model.predict(source="https://youtu.be/4aWufTZDLMU?si=Vd24h_w39XJb1PgX", save=True)
    # model.predict(source="tcp://127.0.0.1:23000", save=True, name="ov_stream")

    suffix, fourcc = (".avi", "MJPG")
    vid_writer = cv2.VideoWriter(
        filename=str(Path(save_path).with_suffix(suffix)),
        fourcc=cv2.VideoWriter_fourcc(*fourcc),
        fps=30,  # integer required, floats produce error in MP4 codec
        frameSize=(int(width), int(height)),  # (width, height)
    )

    idx = 0
    sec = 40
    fps = 30
    # Reset video duration
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened():
        # Read a frame from the video
        idx += 1
        if idx > fps * sec:
            break

        success, frame = cap.read()

        if not success:
            break

        if idx % fps == 0:
            print(f"{idx // fps}/{sec}")
            print(frame.shape)

        # Run YOLO inference on the frame

        # frame = torch.tensor(frame).transpose(0, -1).unsqueeze(0)
        pre_frame = transform_fn(frame)
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
