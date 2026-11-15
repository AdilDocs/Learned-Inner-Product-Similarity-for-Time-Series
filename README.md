# Revised submission package


## code/
- `step1.py` to `step17_*.py`: the original pipeline, unchanged. `step6.py` now also reads the archive location from the `UCR_DIR` environment variable.
- `step18_batched_encoder.py`: batched version of the original encoder with a hand-written backward pass. Its gradient matches `step4` to about 1e-17.
- `step19_dtw_baselines.py`: exact 1-NN DTW and cDTW with LB_Kim, LB_Keogh and early abandoning (Numba); learns the window by leave-one-out cross-validation; also provides embedding-ordered exact search.
- `step20_full_archive.py`: the original encoder against all baselines on all 85 datasets (results in `results/original_encoder/`).
- `step21_stats_figures.py`: statistics for the step20 run.
- `step22_encoder_v2.py`: the revised encoder (PyTorch).
- `step23_dev_selection.py`: design selection on the training splits of 16 random development datasets (no test data).
- `step24_final_evaluation.py`: one-time evaluation of the frozen design (`frozen_config.json`) on all 85 test sets.
- `step25_starlight_native.py` and `step27_starlight_ordered.py`: the editor's StarLightCurves example at native length.
- `step26_paper_numbers.py`: generates `paper/numbers.tex`, the tables and the figures. Run it from `code/`. It expects `final_results_*.csv`, `dev_selection_*.csv`, `results_v1/full_archive_results_*.csv`, `starlight_native.json` and `starlight_ordered.json` in that folder; copy them in from `results/`.
- `step28_tie_check.py`: confirms that ordered and standard cDTW search differ only on exact distance ties.

Requirements: numpy, scipy, pandas, matplotlib, numba, torch.

To reproduce:
```
export UCR_DIR=/path/to/UCR_TS_Archive_2015
python step23_dev_selection.py 0 1
python step24_final_evaluation.py 0 1
python step25_starlight_native.py
python step27_starlight_ordered.py
python step26_paper_numbers.py
```

## results/
- `final_results_all85.csv`: per-dataset accuracy, costs and pruning statistics for the revised encoder, ED, DTW, cDTW and ordered cDTW.
- `dev_selection_*.csv`: design-selection runs (training data only).
- `original_encoder/original_encoder_all85.csv`: the original encoder on all 85 datasets.
- `starlight_*.json`: native-length StarLightCurves experiments.
- `final_summary.json`: all aggregate statistics.

Hardware: single core of the build machine; PyTorch 2.15 (CPU), Numba-compiled DTW. One training run per dataset (seed 0).
