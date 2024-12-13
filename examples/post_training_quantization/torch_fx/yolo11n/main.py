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
import sys
from pathlib import Path
from time import time
from typing import Iterator

os.environ["TORCHINDUCTOR_FREEZING"] = "1"

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

OUTPUT_VIDEO_LEN_SEC = 40
SUBSET_SIZE = 300


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
        subset_size=SUBSET_SIZE,
        preset=nncf.QuantizationPreset.MIXED,
        model_type=nncf.ModelType.TRANSFORMER,
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
def main(quantize_model: bool, async_: bool):
    model = YOLO(ROOT / f"{MODEL_NAME}.pt")

    # Open the video file
    video_path = sys.argv[1]

    save_path = "out_int8" if quantize_model else "out"
    save_path += "_async" if async_ else "_sync"
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
            if quantize_model:
                pt_model = torch.export.export(pt_model, args=(transform_fn(np_dummy_tensor),), strict=False).module()
            else:
                pt_model = torch.export.export(pt_model, args=(transform_fn(np_dummy_tensor),)).module()
            # pt_model = torch.compile(pt_model, backend="openvino")

    if quantize_model:
        pt_model = quantize(pt_model, CV2VideoDataset(cap), transform_fn)
        # pt_model = torch.export.export(pt_model, args=(torch.ones((1, 3, 384, 640)),), strict=False).module()
        pt_model = torch.compile(pt_model, backend="openvino")
    else:
        # pt_model = torch.compile(pt_model, backend="openvino")
        pt_model = torch.compile(pt_model)
    # JIT
    with torch.no_grad():
        with disable_patching():
            pt_model(transform_fn(np_dummy_tensor))
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

    with torch.no_grad():
        with disable_patching():
            if async_:
                inputs = transform_fn(np_dummy_tensor)
                exported_model = torch.export.export(pt_model, args=(inputs,))
                ov_model = ov.convert_model(exported_model, example_input=inputs)
                compiled_model = ov.Core().compile_model(ov_model)
                res = run_async(cap, model, compiled_model, transform_fn, vid_writer, width, height)
            else:
                res = run_sync(cap, model, pt_model, transform_fn, vid_writer)

    # Release the video capture object and close the display window
    cap.release()
    vid_writer.release()

    cap = cv2.VideoCapture(str(Path(save_path).with_suffix(suffix)), cv2.CAP_FFMPEG)

    new_fps = int(1 / res.avg)
    vid_writer = cv2.VideoWriter(
        filename=str(Path(save_path + "_actual_speed").with_suffix(suffix)),
        fourcc=cv2.VideoWriter_fourcc(*fourcc),
        fps=new_fps,  # integer required, floats produce error in MP4 codec
        frameSize=(int(width), int(height)),  # (width, height)
    )

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if not success:
            break
        vid_writer.write(frame)

    cap.release()
    vid_writer.release()
    print(f"quantize=={quantize_model}")
    print(res.__dict__)
    return res


def run_sync(cap, model, pt_model, transform_fn, vid_writer):
    idx = 0
    sec = OUTPUT_VIDEO_LEN_SEC
    fps = 30
    # Reset video duration
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    average_meter = AverageMeter()
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

        # Run YOLO inference on the frame

        # frame = torch.tensor(frame).transpose(0, -1).unsqueeze(0)
        start = time()
        pre_frame = transform_fn(frame)
        results = pt_model(pre_frame)
        results = model.predictor.postprocess(results, pre_frame, [frame])

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        lat = time() - start
        average_meter.update(lat)

        vid_writer.write(annotated_frame)

    return average_meter


def run_async(cap, model, compiled_model, transform_fn, vid_writer, width, height):
    idx = 0
    sec = OUTPUT_VIDEO_LEN_SEC
    fps = 30
    # Reset video duration
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    res_video = np.empty((int(sec * fps), int(height), int(width), 3), dtype=np.uint8)
    average_meter = AverageMeter()

    def callback(infer_request, info) -> None:
        res = infer_request.get_output_tensor(0).data[0]
        res = list(map(torch.tensor, (t.data for t in infer_request.output_tensors)))
        res = [res[0], res[1:]]
        results = model.predictor.postprocess(res, info[1], [info[2]])
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # lat = time() - info[-1]
        # average_meter.update(lat)
        res_video[info[0]] = annotated_frame

    infer_queue = ov.AsyncInferQueue(compiled_model, 8)
    infer_queue.set_callback(callback)

    start = time()
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

        # Run YOLO inference on the frame

        # frame = torch.tensor(frame).transpose(0, -1).unsqueeze(0)
        pre_frame = transform_fn(frame)
        infer_queue.start_async(pre_frame, (idx, pre_frame, frame))

    average_meter.avg = (time() - start) / res_video.shape[0]

    for img in res_video:
        vid_writer.write(img)

    return average_meter


if __name__ == "__main__":
    fp32_res = main(quantize_model=False, async_=False)
    int8_res = main(quantize_model=True, async_=False)
    async_int8_res = main(quantize_model=True, async_=True)
    print(f"fp32: {fp32_res.__dict__}")
    print(f"int8: {int8_res.__dict__}")
    print(f"int8: {async_int8_res.__dict__}")
    print(f"avg speedup with quantization: {fp32_res.avg / int8_res.avg :.3f}")
    print(f"avg speedup with async: {int8_res.avg / async_int8_res.avg :.3f}")
