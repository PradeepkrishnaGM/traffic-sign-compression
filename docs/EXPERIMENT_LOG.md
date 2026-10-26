# Experiment log

## Setup
- Hardware: one desktop CPU (Intel i5-10500T, 6 cores, AVX2, no VNNI), 16 GB RAM, no GPU.
- Software: Python 3.14, PyTorch 2.14 (CPU), torchvision, ONNX, ONNX Runtime 1.31, OpenCV, NumPy, pandas, matplotlib (`requirements.txt`).
- Data: GTSRB (26,640 training, 12,630 test images), resized to 64x64 and cached as uint8 arrays. Validation is split by track, about 10% of the tracks per class.
- All tunable numbers are in `src/config.py`. Seeds are fixed. Corruptions are seeded per image, so every corrupted test set can be rebuilt exactly.

## Steps and durations

### v1 (first full run)
A compute check first timed one teacher batch: 0.76 s per batch of 128 images, or about 2.4 min per epoch. The projected total was below 8 hours, so no epochs were reduced.

| Step | Script | Time |
|---|---|---|
| Data cache, sample figures | `00_explore.py` | seconds |
| Teacher, 10 epochs | `01_train_baselines.py` | 24.0 min |
| Student, 30 epochs | `01_train_baselines.py` | 64.0 min |
| ONNX export, INT8, accuracy check | `02_quantize.py` | about 1 min |
| Pruning 50/70/90%, 3 fine-tuning epochs each | `03_prune.py` | 21.7 min of fine-tuning |
| Distilled student, 30 epochs | `04_distill.py` | 77.3 min |
| INT8 and weather-calibrated INT8 (6 models) | `04_distill.py` | a few minutes |
| Corrupted test sets, inference of 12 variants x 31 conditions | `05_robustness.py` | 48.2 min of inference |
| Latency, figures | `06_results.py` | a few minutes |

Total: about 3.9 hours.

### v2 (teacher without max-pool)
Teacher training was slower with the larger feature maps: 1.42 s per batch, or about 4.4 min per epoch.

| Step | Script | Time |
|---|---|---|
| Teacher, 10 epochs | `01_train_baselines.py` | 48.4 min |
| ONNX export, INT8 | `02_quantize.py` | about 2 min |
| Pruning 50/70/90%, 3 fine-tuning epochs each | `03_prune.py` | 49 min |
| Distilled student, 30 epochs, INT8 variants | `04_distill.py` | 109 min (107.6 min training) |
| Inference of the 9 teacher-dependent variants x 31 conditions | `05_robustness.py` | 93 min |
| Latency, figures | `06_results.py` | a few minutes |

Total: about 5.1 hours. The scratch student and its INT8 variants were not retrained. Their code and files did not change.

### Fog follow-up (exploratory)
| Step | Script | Time |
|---|---|---|
| 6 extra INT8 variants (first conv, final linear, or both kept in FP32, for both students), evaluated on 31 conditions | `07_fog_ablation.py` | 21 min |
| Latency of these models, per-layer output error (500 clean vs 500 fog-5 images) | `08_fog_diagnostics.py` | a few minutes |
| Simulated rounding of the first layer's input and weights in PyTorch | `09_first_conv_split.py` | about 2 min |
| Fused vs unfused ONNX Runtime execution, U8U8 weights | `10_ort_fusion_check.py` | about 5 min |

### INT8 correction
| Step | Script | Time |
|---|---|---|
| `reduce_range=True` vs old default vs U8U8 on the plain INT8 student | `11_reduce_range_test.py` | about 3 min |
| Re-quantization of the 6 INT8 variants | `12_requantize_int8.py` | about 1 min |
| Inference of the 6 new variants x 31 conditions, metrics for 18 variants | `05_robustness.py` | 31 min |
| Latency of 18 variants, results table, figures | `06_results.py` | about 5 min |

### Additional metrics
Error ratios and flip rates restricted to images the FP32 parent got right were added to `results/results.csv` from saved logits (`13_relative_metrics.py`). No inference was run and the latencies were not re-measured.

## Deviations from the initial design
- CPU scripts instead of notebooks. The student uses 64x64 input, the same as the teacher.
- 12 variants instead of 8. Added: INT8 for the scratch student, and a weather-calibrated INT8 variant for the teacher, the student and the distilled student.
- Images are resized whole, without cropping to the annotated sign region.
- Spearman correlation is computed with NumPy, because SciPy is not used.
- PyTorch latency is reported only for the FP32 models, since no PyTorch INT8 models exist.
- Structured channel pruning was not done.
- No hyper-parameter was tuned. The defaults in `src/config.py` were used once.
- The teacher architecture changed between v1 and v2 (see below).
- INT8 uses `reduce_range=True` instead of ONNX Runtime's default (see below). The old INT8 results are kept as a separate group.

## Problems found and how they were fixed

### Weak teacher (max-pool)
The v1 teacher reached 96.90%, below the 97% target, and the small students beat it on clean data. ResNet-18 reduces its input 4x before the first block: a stride-2 convolution, then max-pooling. A 64x64 image therefore leaves only 16x16 feature maps. v2 replaces the max-pool layer with `nn.Identity()` and keeps the first convolution, the ImageNet weights and the training settings. The teacher reached 98.27% clean and 78.32% under corruption (v1: 96.90% and 77.70%). Its latency rose from 5.94 ms to 12.35 ms. Everything that depends on the teacher was re-run.

Effects on the conclusions:
- In v1 the scratch student beat the teacher on clean data. In v2 it does not (0.33 points below). The ranking reversal now holds only for the distilled student.
- The pruning result is stable. At 90% sparsity, the drop under corruption is 3.9 times the clean drop in v1 and 2.8 times in v2.

### Corruption difficulty
The first corruption grid showed low light at severity 5 almost black and fog at severity 5 almost white. Severity 5 is meant to be hard but still recognisable. Fog strength and the low-light gain and gamma were reduced in `src/corruptions.py`. The corrupted sets, all predictions and the weather-calibrated models were then rebuilt. All reported numbers use the final corruption code.

### INT8 saturation in ONNX Runtime
With ONNX Runtime's default INT8 setting, the INT8 students lost 3.8 to 4.0 points on clean data and up to 35 points in fog at severity 5. The INT8 teacher lost much less. The fog follow-up tested whether low contrast plus first-layer rounding was the cause. That hypothesis was not supported:
- Keeping only the first convolution in FP32 restored accuracy on clean data and on all corruptions (student: 94.17% to 97.91% clean). The final linear layer made no difference.
- Simulated 8-bit rounding of the first layer changed accuracy by at most 0.03 points.
- Per-layer output errors were small (about 2 to 6% of the activation size). No layer showed a jump.
- The same INT8 file reached FP32-level accuracy when ONNX Runtime ran it unfused (97.94% clean), and also with unsigned weights (U8U8).

The ONNX Runtime quantization documentation (section "Data type selection") explains the cause. On x86-64 CPUs with AVX2 or AVX512 but without VNNI, the U8S8 kernel uses `VPMADDUBSW`. That instruction adds pairs of products in 16 bits, and the sums can saturate. The documented remedies are `reduce_range` or U8U8.

The rule fixed before the test was: use `reduce_range` if it gives FP32-level accuracy at INT8 speed, otherwise U8U8. Results on the plain INT8 student:

| setting | clean % | fog severity 5 % | latency ms |
|---|---|---|---|
| FP32 | 97.94 | 68.04 | 2.84 |
| INT8, old default | 94.17 | 37.26 | 1.94 |
| INT8, `reduce_range=True` | 97.89 | 67.94 | 1.97 |
| INT8, U8U8 | 97.82 | 67.62 | 2.95 |

`reduce_range=True` became the default (`config.INT8_REDUCE_RANGE`). All six INT8 variants were re-quantized and re-evaluated. Latencies were measured again. The old files and predictions were renamed `*_ortdefault` and form the group "int8_ort_default (saturating, non-VNNI CPU)" in `results/results.csv`. The fog follow-up models and the U8U8 model still use the old default and are labelled as such.

Effects on the conclusions:
- INT8 went from a visible loss to almost no loss.
- The Spearman correlation between clean and corrupted accuracy fell from 0.81 to 0.55.
- Weather-aware calibration remained a null result.

## Checks performed
- Unit tests (`python -m unittest`, quick; `RUN_SLOW=1` for slow ones) cover:
  - no track appearing in both the training and validation splits;
  - deterministic corruptions that get stronger with severity;
  - student size;
  - KD loss with alpha = 0;
  - pruning sparsity;
  - ONNX FP32 output against PyTorch;
  - ECE on hand-computed cases, flip rate and Spearman edge cases;
  - Pareto front ties;
  - the parent-correct flip rate;
  - the composition of the weather calibration set;
  - the teacher having no max-pool layer.
- Every ONNX FP32 model was checked against its PyTorch model for both accuracy and logits.
- The corrupted-set cache and the weather calibration set were checked against the current corruption code.
- Latency was measured with no other job of this project running and total CPU use below 15%. The student latencies agree between v1 and v2 to within 2% (2.87 vs 2.84 ms FP32, 1.97 vs 1.94 ms INT8).
- In v1 the plain and weather-calibrated KD INT8 models had identical accuracy. The two files were checked and differ, so the match was a coincidence.

## Open questions
- Run-to-run variance was not measured (single seed). Differences below about 0.3 to 0.5 points should not be interpreted. Per-class numbers rest on as few as 60 test images per class.
- The better v2 teacher gained 1.37 points on clean data but only 0.62 under corruption, so its robustness gap grew from 19.2 to 20.0 points. The cause was not investigated.
- The 90%-pruned teacher drops to 13.2% accuracy before fine-tuning (96.98% after). The classes the damaged network predicts were not examined.
- The 90%-pruned teacher and the students are better calibrated under corruption than the teacher, because they are less confident in general. It is unclear whether this helps in practice.
- In v1 the distilled student was more over-confident under corruption than the scratch student (ECE 0.087 vs 0.034). In v2 it is not (0.025 vs 0.034). Not investigated.
- Saturation was inferred from the documentation and from the effect of the fix, not measured directly. VNNI hardware and other inference engines were not tested.
- Weather-aware calibration could not be judged, because there was almost no INT8 loss to recover. A harsher setting, for example 4-bit weights, would be needed to test it.
- The flip rates of the FP32 students against the teacher (25 to 26% under corruption) mix compression with an architecture change and are not compression damage alone.
