"""Docs-and-analysis fix: add error ratios and parent-correct flip rates to results.csv, redraw the
Pareto figures. Uses saved logits only (no inference, no latency re-run)."""
import sys
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import analysis, config as C, data

# 06_results.py starts with a digit, so import it by path to reuse its plotting function
spec = importlib.util.spec_from_file_location("results06", Path(__file__).with_name("06_results.py"))
r06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r06)

d = data.load_arrays()
table = pd.read_csv(C.RESULTS_DIR / "results.csv")
table = table.drop(columns=[c for c in table.columns if c.startswith("err_ratio") or c.endswith("parentcorrect")])
# keep the new columns next to the flip-rate columns' end; notes stays last
table = analysis.add_relative_metrics(table, d["test_y"], C.CACHE_DIR / "preds")
cols = [c for c in table.columns if c != "notes"] + ["notes"]
table = table[cols]
table.to_csv(C.RESULTS_DIR / "results.csv", index=False)

main_table = table[table["group"] == "main"].reset_index(drop=True)
r06.pareto_figure(main_table, "size_MB", "model file size (MB, ONNX, log scale)", "pareto_size.png")
r06.pareto_figure(main_table, "latency_ms_cpu", "CPU latency (ms, batch 1, 1 thread, log scale)", "pareto_latency.png")
show = ["variant", "group", "err_ratio_clean", "err_ratio_corr", "flip_rate_clean", "flip_rate_clean_parentcorrect",
        "flip_rate_corr", "flip_rate_corr_parentcorrect"]
print(table[show].round(3).to_string())
