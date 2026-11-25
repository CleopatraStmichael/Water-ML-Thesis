# Can machine learning predict what response-surface models couldn't?

**Re-modelling a 31-run adsorption experiment with Gaussian processes and random forests**

Cleopatra Folakemi Stmichael · B.Sc. Petroleum and Gas Engineering, University of Lagos (2026)

## The question

In my thesis, activated sugarcane-peel and castor-leaf adsorbents removed more than 94% of Pb, Cd and Ni from real produced water. However, the quadratic response-surface (RSM) models were not significant and had low predictive R², so I did not claim an optimum.

This project asks: **was the problem the model's shape, or is there not enough signal in 31 runs?**

- Quadratic RSM assumes smooth curves, one bend per factor.
- A **Gaussian process (GP)** can bend more flexibly and gives an uncertainty for each prediction.
- A **random forest (RF)** can capture thresholds and interactions.

If all three fail on held-out runs, the limit is the data (noise and the narrow range of removals), not the method. That is still a useful result, because it tells you what the next experiments must change.

## Method

- Data: 31 runs of a 4-factor Central Composite Design. The factors are castor-leaf mass, sugarcane-peel mass, contact time and temperature. The responses are % removal of Pb, Cu, Cr, Ni, Cd and Fe.
- Models: mean-only baseline, quadratic RSM, GP (RBF kernel with one length scale per factor, plus a noise term), and random forest.
- Validation: **leave-one-out cross-validation**. Each run is predicted by a model that never saw it. The project reports:
  - **Q²** (predictive R²). Above 0.5 is useful; below 0 is worse than guessing the average.
  - **RMSE** in % removal.
  - The RSM's in-sample R², to show the gap between fitting the data and predicting new runs.
- Factor importance: GP length scales (shorter means a more influential factor) and RF importances.

## Results

**1. No model predicted removal from the experimental factors.** Every model scored Q² below 0 on held-out runs for all six metals. That means each did worse than simply guessing the average removal.

| Metal | Quadratic RSM Q² (in-sample R²) | Gaussian process Q² | Random forest Q² | Run order only Q² |
|---|---|---|---|---|
| Pb | −3.66 (0.25) | −0.09 | −0.19 | −0.14 |
| Cu | −0.74 (0.46) | −0.15 | −0.25 | **0.32** |
| Cr | −1.21 (0.26) | −0.13 | −0.30 | 0.07 |
| Ni | −1.15 (0.46) | −0.21 | −0.21 | 0.09 |
| Cd | −0.97 (0.50) | −0.18 | −0.43 | **0.37** |
| Fe | −0.56 (0.60) | −0.17 | −0.20 | −0.01 |

The quadratic RSM fitted the data moderately (R² 0.25–0.60) but predicted unseen runs far worse than the mean. This is classic overfitting: 15 terms estimated from 31 noisy runs. The more flexible GP and random forest did not help, so **the model's shape was not the limiting factor.**

**2. Identical runs gave very different results.** The 7 centre-point replicates were run at identical settings, yet for Cu, Cr and Fe they varied as much as or more than the whole design did:

| Metal | Centre replicates, range (%) | SD centre / SD all runs |
|---|---|---|
| Cu | 49.9 – 99.2 | 1.27 |
| Cr | 19.9 – 97.0 | 1.49 |
| Fe | 34.2 – 96.8 | 0.93 |
| Cd | 87.4 – 98.8 | 1.05 |
| Ni | 88.5 – 97.6 | 0.93 |
| Pb | 96.4 – 99.2 | 0.57 |

When repeating the same conditions varies this much, no model can detect the effect of changing the conditions.

**3. Removal drifted upward with the order in which runs were performed.** Run order correlated with removal (Spearman ρ): Cd ρ = 0.66 (p < 0.001), Cu ρ = 0.49 (p = 0.005) and Ni ρ = 0.44 (p = 0.012). On its own, the sequence number predicted Cu and Cd removal (Q² ≈ 0.3–0.4) better than all four experimental factors combined. The run-order chart shows a step change: removals were lower and more erratic in roughly the first 10 runs, then stabilised.

**4. Iron behaved differently.** In run 6, Fe removal was −26%, so the solution held more iron after treatment than before. The plant-based adsorbent may have released iron.

**Conclusion:** within the ranges tested, the effects of adsorbent dose, time and temperature were smaller than the run-to-run variability. A time-related drift during the experiment, rather than the choice of model, is the most likely main source of that variability.

## Limitations

- 31 runs is a small dataset. Leave-one-out validation measures overfitting but cannot overcome it.
- Run order is a stand-in for whatever changed over time. The data cannot say *what* changed. Possible causes include AAS calibration drift, adsorbent ageing or moisture uptake, storage time of samples before analysis, or improving technique.
- The design's run order was not fully randomised against time-dependent effects. Because of this, drift can be confused with the effects of the factors.
- Results apply to one produced-water sample and these two adsorbents.
- Run 10's sugarcane-peel mass was set to 0 g. The design called for a negative axial value (−0.04 g), which is not physically possible.

## What this means for the next experiments

1. **Control noise first.** Use calibration check standards and blanks during every AAS session, and record dates and batches.
2. **Randomise and block.** Randomise the run order, and record the day and adsorbent batch so drift can be modelled as a block effect.
3. **Spread replicates through the sequence.** Space the centre-point replicates across the run order, and run more of them, to track drift.
4. **Prepare one adsorbent batch and condition it** before starting, then store it in a desiccator.
5. **Investigate iron release** from the bio-adsorbents. It matters for treated-water quality, especially if the adsorbents are used for drinking water.
6. Only once replicate noise is small, return to modelling. The GP's uncertainty estimates can then guide which runs to add.

---

## How to run it

**Easiest, with nothing to install:** open `compare_models_colab.ipynb` in [Google Colab](https://colab.research.google.com). Upload the notebook, `compare_models.py` and your data CSV, then run each cell.

**On your own computer:**

```bash
pip install -r requirements.txt
python compare_models.py --demo                      # check it works (synthetic data)
python compare_models.py --data data/thesis_runs.csv # your real data
```

Results go to `results/`:

- `summary.csv`: Q², RMSE and RSM in-sample R² for each metal and model
- `factor_importance.csv`
- `parity_<metal>.png`: predicted vs. measured for each held-out run
- `q2_comparison.png`
- `diagnostics.csv`: centre-replicate noise and run-order drift for each metal
- `run_order_and_replicates.png`: removal plotted against the order runs were performed

`GP_hit_limit = True` means the GP pushed a length scale to its limit. This usually means that factor has almost no effect on that metal, which is worth reporting, not an error.

## Data format

Copy `data/thesis_runs_TEMPLATE.csv` to `data/thesis_runs.csv` and fill one row per run, using real units (not coded −2…+2 values):

| Column | Meaning |
|---|---|
| `run` or `RunOrder` | Order the runs were actually performed in (used for the drift check) |
| `point_type` or `PtType` | factorial / axial / center, or Design-Expert's 1 / −1 / 0 (used for the replicate check) |
| `castor_g` | Castor-leaf mass (g) |
| `peel_g` | Sugarcane-peel mass (g) |
| `time_min` | Contact time (min) |
| `temp_C` | Temperature (°C) |
| `removal_Pb` … `removal_Fe` | % removal for each metal |

You can leave a metal's column blank if you don't have it. Any column starting with `removal_` is modelled automatically.
