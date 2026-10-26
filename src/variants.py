"""The 12 compared model variants in one place (id -> file, parent, names used in tables)."""
from collections import OrderedDict

# id: (model family, method name in results.csv, onnx file, FP32 parent id or None)
VARIANTS = OrderedDict([
    ("teacher_fp32",         ("teacher", "fp32",                "teacher_fp32.onnx", None)),
    ("teacher_int8",         ("teacher", "int8_static",         "teacher_int8.onnx", "teacher_fp32")),
    ("teacher_int8_wcal",    ("teacher", "int8_static_wcal",    "teacher_int8_wcal.onnx", "teacher_fp32")),
    ("teacher_prune50",      ("teacher", "prune_50",            "teacher_prune50_fp32.onnx", "teacher_fp32")),
    ("teacher_prune70",      ("teacher", "prune_70",            "teacher_prune70_fp32.onnx", "teacher_fp32")),
    ("teacher_prune90",      ("teacher", "prune_90",            "teacher_prune90_fp32.onnx", "teacher_fp32")),
    ("student_fp32",         ("student", "fp32_scratch",        "student_fp32.onnx", "teacher_fp32")),
    ("student_int8",         ("student", "int8_static",         "student_int8.onnx", "student_fp32")),
    ("student_int8_wcal",    ("student", "int8_static_wcal",    "student_int8_wcal.onnx", "student_fp32")),
    ("student_kd_fp32",      ("student", "fp32_kd",             "student_kd_fp32.onnx", "teacher_fp32")),
    ("student_kd_int8",      ("student", "kd_int8",             "student_kd_int8.onnx", "student_kd_fp32")),
    ("student_kd_int8_wcal", ("student", "kd_int8_static_wcal", "student_kd_int8_wcal.onnx", "student_kd_fp32")),
])

# PyTorch checkpoint for the FP32-based variants (used for the PyTorch latency reference and parameter counts)
TORCH_CKPT = {
    "teacher_fp32": ("teacher", "teacher_fp32.pt"),
    "teacher_prune50": ("teacher", "teacher_prune50.pt"),
    "teacher_prune70": ("teacher", "teacher_prune70.pt"),
    "teacher_prune90": ("teacher", "teacher_prune90.pt"),
    "student_fp32": ("student", "student_fp32.pt"),
    "student_kd_fp32": ("student", "student_kd_fp32.pt"),
}


# The same six INT8 variants quantized the OLD way (ONNX Runtime default, full 8-bit signed weights).
# On this AVX2 CPU without VNNI that setting saturates silently (see README, follow-up). Kept as a separate group.
LEGACY = OrderedDict()
for _id, (_family, _method, _file, _parent) in list(VARIANTS.items()):
    if "int8" in _id:
        LEGACY[_id + "_ortdefault"] = (_family, _method + "_ortdefault", _file.replace(".onnx", "_ortdefault.onnx"), _parent)

ALL_VARIANTS = OrderedDict(list(VARIANTS.items()) + list(LEGACY.items()))
GROUP = {**{k: "main" for k in VARIANTS}, **{k: "int8_ort_default (saturating, non-VNNI CPU)" for k in LEGACY}}
