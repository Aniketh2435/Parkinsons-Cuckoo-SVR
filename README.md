# Parkinson's Disease Severity Prediction from Voice Recordings using SVR Optimised with Cuckoo Search

Bio-Inspired Algorithms AAT (23DS7BSBIO), Group 15, Dept. of CSE (Data Science), BMSCE.

An RBF-kernel Support Vector Regressor predicts a patient's motor UPDRS score from 18 voice and
demographic features. **Cuckoo Search**, implemented from scratch, jointly tunes the SVR's
hyperparameters (C, gamma, epsilon) and selects the input-feature subset. Random Search with the
same budget is the non-bio-inspired reference. Before the SVR experiment, the Cuckoo Search operators are
checked on four standard test functions (Sphere, Rastrigin, Rosenbrock, Ackley).

**Team:** Abhijith M Reddy (1BM23CD003), Aniketh S Nadig (1BM23CD006), Mayur M (1BM23CD037),
S Sanjay (1BM23CD053), Tanush V Gowda (1BM23CD064). **Guide:** Prof. Manoj Kumar S.

## Results (held-out test set)
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Default SVR | 5.672 | 4.411 | 0.496 |
| Random Search SVR | 4.693 | 3.297 | 0.655 |
| **Cuckoo-SVR** | **4.347** | **3.066** | **0.704** |

Over 10 seeds: Cuckoo Search 4.356 ± 0.042 vs Random Search 4.596 ± 0.186 (Cuckoo wins 8/10, Wilcoxon p = 0.005).

## Deliverables
- `report/Group15_Cuckoo_SVR_Report.pdf`: the AAT report (LaTeX source: `report/main.tex`)
- `presentation/Group15_Cuckoo_SVR_Presentation.pptx` / `.pdf`: the slides

## Proof of the run
- `Parkinsons_Cuckoo_SVR.ipynb`: the executed notebook (code, logs, tables, plots; run date and machine at the top)
- `Parkinsons_Cuckoo_SVR.html`: the same notebook, viewable in any browser
- `outputs/`: every figure, metric CSV, both search logs, `results.json` and the trained model

## Setup and re-run
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Parkinsons_Cuckoo_SVR.ipynb --ExecutePreprocessor.timeout=-1
```
The full run takes about 3 hours on an Apple M4 CPU (10 cores). All randomness is seeded, so results are reproducible.

## Data
UCI *Parkinson's Telemonitoring* (Tsanas et al., 2009): `data/parkinsons_updrs.data`, with the
description in `data/parkinsons_updrs.names`. 5,875 recordings from 42 patients.

## Project layout
| File | Purpose |
|---|---|
| `src/data.py` | Cleaning, log transform of skewed voice measures, 80/20 split |
| `src/model.py` | SVR pipeline (feature + target scaling) and the untuned baselines |
| `src/search_space.py` | Nest encoding (3 log-scaled hyperparameters + 18 feature bits) and the CV fitness function |
| `src/cuckoo.py` | Cuckoo Search from scratch: Levy flights, nest replacement, abandonment, elitism, warm start |
| `src/benchmarks.py` | Sphere, Rastrigin, Rosenbrock and Ackley test functions for checking the optimiser |
| `src/random_search.py` | Random Search reference optimiser |
| `src/metrics.py` | RMSE, MAE, R^2, MAPE and before/after comparison tables |
| `src/plots.py` | All report figures |
| `build_notebook.py` | Generates `Parkinsons_Cuckoo_SVR.ipynb` (run with the notebook path as argument) |

`archive/` holds
`run_v1_random_replacement/`: the first full Parkinson's run, made before the benchmark check showed that
letting a new egg replace a *random* nest makes the population collapse. The current code keeps the egg
only if it beats its own nest, as in Yang & Deb's reference implementation.
`run_v2_budget170/`: the run with the fixed algorithm but only 170 evaluations per search, where Cuckoo Search
and Random Search tied (4/10 wins). `budget_study_curves.json` holds the 3-seed budget study (about 1,000
evaluations), which showed Cuckoo Search pulling ahead after about 500 evaluations; the budget was then raised
to about 820 evaluations for both methods.
