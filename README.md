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
- Data-quality checks: spread of the centre-point replicates, correlation with run order, and control charts (Shewhart individuals chart with ±3σ limits from the average moving range; flags points outside the limits and 8 or more consecutive points on one side of the mean).
- Uncertainty: leave-one-out 95% prediction intervals from the GP, and how often they contain the measured value.
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

**3. Removal drifted upward with the order in which runs were performed.** Run order correlated with removal (Spearman ρ): Cd ρ = 0.66 (p < 0.001), Cu ρ = 0.49 (p = 0.005) and Ni ρ = 0.44 (p = 0.012). On its own, the sequence number predicted Cu and Cd removal (Q² ≈ 0.3–0.4) better than all four experimental factors combined. The run-order chart shows a step change: removals were lower and more erratic in the first 10 runs, then stabilised.

**4. Iron behaved differently.** In run 6, Fe removal was −26%, so the solution held more iron after treatment than before. The plant-based adsorbent may have released iron.

**5. A control chart pinpoints when the process changed.** Individuals (Shewhart) control charts of removal against run order flag the same period for several metals:
- **Cu:** runs 1, 3, 4, 8, 9 and 10 fall below the lower 3σ limit.
- **Cr:** runs 1, 2 and 6 fall below the limit.
- **Cd:** runs 1, 3 and 9 fall below the limit. From run 11 onward, all 21 remaining runs sit above the centre line, which is a clear shift.

The process behaved differently in the first 10 runs, then settled.

**6. The Gaussian process "knows what it doesn't know".** For each held-out run, the GP gives a prediction with a 95% interval. Those intervals contained 87–97% of the measured values, so the uncertainty is honest. However, the intervals were very wide: about ±4% for Pb and ±7–8% for Ni and Cd, but ±32–48% for Cu, Cr and Fe. The model fell back to roughly the average and widened its uncertainty instead of making confident wrong predictions. That is the behaviour you want from a model used for safety-relevant decisions.

**7. The shift lines up with a change of day.** Runs 1–10 were performed on the first day, and run 11 began a new day, exactly where the control charts show the shift. Comparing day 1 with the later runs:

| Metal | Day 1 mean (SD) | Later mean (SD) | p (Mann-Whitney) | Later centre replicates |
|---|---|---|---|---|
| Cu | 66.9% (15.1) | 95.7% (5.2) | < 0.001 | 96.5–99.2% |
| Cd | 91.5% (4.3) | 98.3% (0.7) | < 0.001 | 97.3–98.8% |
| Pb | 95.8% (2.3) | 97.8% (2.0) | 0.015 | 96.5–99.2% |
| Ni | 92.4% (4.6) | 95.1% (3.4) | 0.10 | 89.8–97.6% |
| Cr | 72.6% (28.8) | 90.6% (7.6) | 0.13 | 75.4–97.0% |
| Fe | 43.0% (28.6) | 67.9% (20.2) | 0.054 | 49.5–96.8% |

After day 1, the centre replicates for Cu, Cd and Pb agree closely (e.g., Cu within 3 percentage points, against 50–99% overall). So the large replicate noise was mostly a **day 1 effect**, a "block effect" in experimental-design terms. The design was not blocked by day (all runs are in one block), so this effect was confused with the effects of the factors. Cr and Fe stayed noisy in the later runs as well.

**8. Even the stable runs could not be predicted, because removal was near its ceiling.** When the GP and random forest were re-tested on runs 11–31 alone, Q² was still at or below about 0 for every metal (best: random forest for Cu, 0.16). In those runs, removal of Cd (97–100%), Pb (93–99%) and Ni (89–99%) was close to 100% across almost all conditions. When nearly every run removes nearly everything, there is little difference left for the factors to explain. The adsorbent doses were large relative to the low metal concentrations (about 1–2 mg/L), so the design ranges probably sat on a performance plateau.

**Conclusion:** the models failed for two reasons, neither of which a better algorithm could fix. First, a **day-to-day effect** (day 1 vs later) added large variability that the design did not account for. Second, in the stable later runs (from run 11), removal was **close to 100% under nearly all conditions**, so the factors had little left to explain within the ranges tested.

## Relevance to the water industry

This is a small lab dataset, not plant data. Even so, the project works through three problems that decide whether AI can be trusted at a real treatment plant:

1. **Check the data before modelling it.** Treatment plants run on sensors (turbidity, chlorine, pH) that drift as they foul or lose calibration. A model trained on drifting data learns the drift, not the process. This project found the same problem, run-order drift and noisy replicates, before any model could be trusted. The control charts used here are the same tool utilities use to catch sensor faults and process upsets.
2. **A model must beat a simple baseline.** A utility should not use a model that cannot beat "use the average" or "tomorrow will be like today". Every model here was tested against a baseline on unseen runs, and the result was reported honestly when they failed.
3. **Report uncertainty, not just a number.** A confidently wrong prediction about water safety is dangerous. The Gaussian process reports a 95% interval for each prediction. Here those intervals were honest (they covered 87–97% of results) and wide, signalling that the data could not support confident predictions.

The practical lesson for applying AI to water treatment is that data quality and validation come before model complexity.

## Limitations

- 31 runs is a small dataset. Leave-one-out validation measures overfitting but cannot overcome it.
- The control-chart limits were calculated from all 31 runs, including the unstable early period. In a plant, limits would be set from a known stable period and then applied to new data.
- The shift coincides with a change of day, but the data cannot say *what* changed between days. Possible causes include AAS recalibration, a freshly prepared or differently conditioned adsorbent, how long samples were stored before analysis, or room temperature.
- The design's run order was not fully randomised against time-dependent effects. Because of this, drift can be confused with the effects of the factors.
- Results apply to one produced-water sample and these two adsorbents.
- Run 10's sugarcane-peel mass was set to 0 g. The design called for a negative axial value (−0.04 g), which is not physically possible.

## What this means for the next experiments

1. **Control noise first.** Use calibration check standards and blanks during every AAS session, and record dates and batches.
2. **Block by day.** Central Composite Designs can be split into blocks, so each day gets a balanced set of runs including its own centre points. Then a day effect can be estimated and removed instead of being confused with the factors. Also randomise the run order within each day.
3. **Spread replicates through the sequence.** Space the centre-point replicates across the run order, and run more of them, to track drift.
4. **Prepare one adsorbent batch and condition it** before starting, then store it in a desiccator.
5. **Investigate iron release** from the bio-adsorbents. It matters for treated-water quality, especially if the adsorbents are used for drinking water.
6. **Move off the plateau.** Use higher metal concentrations or lower adsorbent doses (an isotherm study would show where) so the conditions actually produce different removals.
7. Only once replicate noise is small, return to modelling. The GP's uncertainty estimates can then guide which runs to add.

---




