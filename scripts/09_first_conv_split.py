"""Extra diagnostic (exploratory): is the first-conv damage from the INPUT rounding or the WEIGHT rounding?

PyTorch simulation on the FP32 student (no ONNX, no retraining):
  input    = the normalised image is rounded to the 256 levels the INT8 input quantizer uses
  weights  = BatchNorm folded into the first conv, then per-channel symmetric INT8 rounding of its weights
  both     = both of the above
The rest of the network stays FP32, so only the first-layer effects are visible (the INT8 activations
of later layers are not simulated). Results: results/first_conv_split.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnx
import pandas as pd
import torch
from onnx import numpy_helper
from torch import nn

from src import config as C
from src import data, models, train


def input_quantizer(int8_path):
    """Scale and zero-point that the INT8 model applies to its input tensor."""
    model = onnx.load(str(int8_path))
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and node.input[0] == "input":
            return float(inits[node.input[1]]), int(inits[node.input[2]])
    raise RuntimeError("input quantizer not found")


def fold_first_conv(net, quantize_weights):
    """Fold BatchNorm into the first conv; optionally round its weights to per-channel INT8."""
    conv, bn = net.features[0][0], net.features[0][1]
    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight * factor.view(-1, 1, 1, 1)
    bias = bn.bias - bn.running_mean * factor
    if quantize_weights:
        scale = weight.abs().amax(dim=(1, 2, 3), keepdim=True) / 127.0       # one scale per output channel
        weight = torch.round(weight / scale).clamp(-127, 127) * scale
    new_conv = nn.Conv2d(3, weight.shape[0], 3, padding=1, bias=True)
    new_conv.weight.data, new_conv.bias.data = weight.detach(), bias.detach()
    net.features[0][0] = new_conv
    net.features[0][1] = nn.Identity()


@torch.no_grad()
def accuracy(net, images, labels, scale, zero_point, quantize_input):
    net.eval()
    correct = 0
    for start in range(0, len(images), 512):
        x = data.to_tensor(np.array(images[start:start + 512]))
        if quantize_input:
            x = (torch.clamp(torch.round(x / scale) + zero_point, 0, 255) - zero_point) * scale
        correct += int((net(x).argmax(1).numpy() == labels[start:start + 512]).sum())
    return correct / len(images)


def main():
    torch.set_num_threads(C.NUM_THREADS)
    d = data.load_arrays()
    scale, zero_point = input_quantizer(C.CKPT_DIR / "onnx" / "student_int8.onnx")
    print(f"input quantizer: step {scale:.5f} (about {scale * np.mean(C.STD) * 255:.2f} grey levels), zero-point {zero_point}", flush=True)
    sets = {"clean": d["test_x"]}
    for s in C.SEVERITIES:
        sets[f"fog_{s}"] = np.load(C.CACHE_DIR / "corrupt" / f"fog_{s}.npy", mmap_mode="r")
    rows = []
    for config, q_in, q_w in (("none", False, False), ("input_only", True, False),
                              ("weights_only", False, True), ("both", True, True)):
        net = models.make_student()
        net.load_state_dict(torch.load(C.CKPT_DIR / "student_fp32.pt"))
        fold_first_conv(net, quantize_weights=q_w)
        row = {"config": config}
        for name, images in sets.items():
            row[name] = accuracy(net, images, d["test_y"], scale, zero_point, q_in)
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "first_conv_split.csv", index=False)


if __name__ == "__main__":
    main()
