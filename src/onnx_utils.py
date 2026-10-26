"""ONNX export, static INT8 quantization and ONNX Runtime inference."""
import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from . import config as C
from . import data


def export_onnx(model, path):
    """Export with a dynamic batch size (evaluation uses big batches, latency uses batch 1)."""
    model.eval()
    dummy = torch.zeros(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    torch.onnx.export(model, dummy, str(path), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)


class CalibrationReader(CalibrationDataReader):
    """Feeds the calibration images one at a time so ORT can record activation ranges."""

    def __init__(self, images_uint8):
        self.batches = iter([{"input": data.to_tensor(images_uint8[i:i + 1]).numpy()}
                             for i in range(len(images_uint8))])

    def get_next(self):
        return next(self.batches, None)


def quantize_int8(fp32_path, int8_path, calib_images_uint8, nodes_to_exclude=None, reduce_range=None):
    """Static INT8 quantization: QDQ format, per-channel weights, activations calibrated on real images.

    Static means activation ranges (scale / zero-point) are measured once on the
    calibration images instead of being computed at run time.

    `nodes_to_exclude` lists node names that stay in FP32 (no quantize/dequantize around them).
    `reduce_range` (default: config.INT8_REDUCE_RANGE) uses 7-bit weights. ONNX Runtime's U8S8 kernel on
    AVX2/AVX512 CPUs without VNNI can overflow (saturate) with full 8-bit weights; 7 bits avoids this.
    """
    if reduce_range is None:
        reduce_range = C.INT8_REDUCE_RANGE
    pre_path = str(fp32_path).replace(".onnx", "_pre.onnx")
    quant_pre_process(str(fp32_path), pre_path)
    quantize_static(pre_path, str(int8_path), CalibrationReader(calib_images_uint8),
                    quant_format=QuantFormat.QDQ, per_channel=True, nodes_to_exclude=nodes_to_exclude,
                    reduce_range=reduce_range,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)


def make_session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def ort_logits(session, images_uint8, batch_size=256):
    out = []
    for start in range(0, len(images_uint8), batch_size):
        # np.array copies out of a memory-mapped cache file, which torch cannot read directly
        x = data.to_tensor(np.array(images_uint8[start:start + batch_size])).numpy()
        out.append(session.run(None, {"input": x})[0])
    return np.concatenate(out)


def calibration_images(train_x, n=C.CALIB_IMAGES, seed=C.SEED):
    """n clean images drawn from the training split (fixed seed)."""
    idx = np.random.RandomState(seed).choice(len(train_x), n, replace=False)
    return train_x[idx]
