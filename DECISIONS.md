# Decisions

- Input size 64x64 for all models, so every corruption is identical for every model.
- Validation split by track (about 10% of tracks per class), not at random, because consecutive frames of one sign are near-duplicates.
- Images are resized whole to 64x64 (no region-of-interest crop), to keep preprocessing simple.
- Everything runs as plain Python scripts on one CPU, without a GPU.
- Compared variants: teacher (FP32, INT8, weather-calibrated INT8, pruned 50/70/90) and student (scratch FP32/INT8/weather-calibrated INT8, KD FP32/INT8/weather-calibrated INT8).
- FP32 parent for drops and flip rate: teacher variants and both FP32 students use the teacher; student INT8 variants use their own FP32 student.
- Weather-calibrated INT8 uses fog, rain and low light only; motion blur, noise and JPEG are held out.
- Spearman correlation is computed with numpy (no scipy).
