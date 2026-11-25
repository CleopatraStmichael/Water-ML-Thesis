"""
Compare a quadratic response-surface model (RSM) with two machine-learning models
(Gaussian process regression and random forest) on the 31-run Central Composite
Design from the thesis "Low-Cost Agricultural-Waste Adsorbents for Heavy-Metal
Removal from Oilfield Produced Water" (C. F. Stmichael, University of Lagos, 2026).

Every model is judged the same way: leave-one-out cross-validation (LOO).
Each run is held out once, the model is trained on the other 30, and it predicts
the one it has never seen. From those predictions we get:
  Q2        = 1 - PRESS / SS_total   (predictive R2; > 0.5 is decent, < 0 is worse than guessing the mean)
  RMSE_LOO  = typical prediction error in % removal
The in-sample R2 of the RSM model is also reported, to show the gap between
"fits the data" and "predicts new runs".

Usage:
  python compare_models.py --data data/thesis_runs.csv
  python compare_models.py --demo        # synthetic data, only to check the code runs
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.stats import spearmanr

FACTORS = ["castor_g", "peel_g", "time_min", "temp_C"]
LABELS = {"castor_g": "Castor-leaf mass", "peel_g": "Sugarcane-peel mass",
          "time_min": "Contact time", "temp_C": "Temperature"}


def make_models():
    """The four models compared. Inputs are standardised inside each pipeline."""
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * RBF(length_scale=[1.0] * len(FACTORS), length_scale_bounds=(1e-1, 1e2))
              + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1e1)))
    return {
        "Mean only (baseline)": DummyRegressor(strategy="mean"),
        "Quadratic RSM": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                                       LinearRegression()),
        "Gaussian process": make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0)),
        "Random forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0),
    }


def demo_data(seed=0):
    """SYNTHETIC 31-run CCD (alpha = 2) on the thesis factor ranges. NOT real results."""
    rng = np.random.default_rng(seed)
    fact = np.array([[a, b, c, d] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1) for d in (-1, 1)])
    axial = np.vstack([s * 2 * np.eye(4)[i] for i in range(4) for s in (-1, 1)])
    coded = np.vstack([fact, axial, np.zeros((7, 4))])
    kind = ["factorial"] * 16 + ["axial"] * 8 + ["center"] * 7
    centre = np.array([0.21, 0.2175, 100.0, 50.0])
    step = np.array([0.105, 0.10875, 38.75, 20.0])
    X = centre + coded * step
    df = pd.DataFrame(X, columns=FACTORS)
    df.insert(0, "point_type", kind)
    df.insert(0, "run", range(1, 32))
    x = coded
    for metal, base, noise in [("Pb", 97, 1.5), ("Cd", 96, 2.0), ("Fe", 60, 20.0)]:
        y = base + 1.5 * x[:, 0] + 1.0 * x[:, 2] + 1.2 * x[:, 0] * x[:, 2] - 0.8 * x[:, 1] ** 2
        df[f"removal_{metal}"] = np.clip(y + rng.normal(0, noise, len(y)), 0, 100).round(2)
    return df


def find_run_order(df):
    """Column holding the order the runs were actually performed in, if any."""
    for c in ["RunOrder", "run_order", "run"]:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().all():
            return c
    return None


def centre_mask(df):
    """True for centre-point replicates (identical conditions), if the file says which they are."""
    if "PtType" in df.columns:
        return pd.to_numeric(df["PtType"], errors="coerce") == 0
    if "point_type" in df.columns:
        return df["point_type"].astype(str).str.strip().str.lower().isin(["center", "centre"])
    return None


def diagnostics(df, out):
    """How noisy are repeated runs, and did results drift over the order runs were done in?"""
    targets = [c for c in df.columns if c.startswith("removal_")]
    order, centre = find_run_order(df), centre_mask(df)
    rows = []
    for t in targets:
        y = df[t]
        row = {"metal": t.replace("removal_", ""), "mean_%": round(y.mean(), 1), "SD_all_runs": round(y.std(), 1)}
        if centre is not None and centre.sum() >= 2:
            cy = y[centre]
            row.update({"n_centre": int(centre.sum()), "SD_centre_replicates": round(cy.std(), 1),
                        "centre_min_%": round(cy.min(), 1), "centre_max_%": round(cy.max(), 1),
                        "noise_ratio": round(cy.std() / y.std(), 2)})
        if order:
            rho, p = spearmanr(df[order], y)
            half = df[order] <= df[order].median()
            row.update({"run_order_rho": round(rho, 2), "run_order_p": round(p, 4),
                        "mean_first_half_%": round(y[half].mean(), 1), "mean_second_half_%": round(y[~half].mean(), 1)})
        rows.append(row)
    diag = pd.DataFrame(rows)
    diag.to_csv(os.path.join(out, "diagnostics.csv"), index=False)

    if order and targets:
        n = len(targets)
        ncol = 3 if n > 2 else n
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
        for ax, t in zip(axes.flat, targets):
            ax.scatter(df[order], df[t], s=22, color="#1f5fa8", label="design runs")
            if centre is not None:
                ax.scatter(df.loc[centre, order], df.loc[centre, t], s=40, color="#d1495b",
                           marker="D", label="centre replicates (identical settings)")
            ax.set_title(t.replace("removal_", ""), fontsize=10)
            ax.set_xlabel("Run order (sequence performed)")
            ax.set_ylabel("Removal (%)")
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
        fig.suptitle("Removal by run order: drift over time and spread of identical centre runs", fontsize=11)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(os.path.join(out, "run_order_and_replicates.png"), dpi=150)
        plt.close(fig)
    return diag


def q2(y, pred):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def evaluate(df, out):
    os.makedirs(out, exist_ok=True)
    missing = [f for f in FACTORS if f not in df.columns]
    if missing:
        raise SystemExit(f"Your file is missing these column headings: {missing}. Check spelling and capitals.")
    for col in FACTORS + [c for c in df.columns if c.startswith("removal_")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    dropped = df[FACTORS].isna().any(axis=1).sum()
    df = df.dropna(subset=FACTORS).reset_index(drop=True)   # e.g. a totals row at the bottom
    if dropped:
        print(f"Note: ignored {dropped} row(s) with missing factor values (e.g. a totals row).")
    for f in FACTORS:
        if (df[f] < 0).any():
            print(f"WARNING: {f} has negative values in runs {list(df.index[df[f] < 0] + 1)}. "
                  "A negative amount is impossible; enter the value actually used (e.g. 0).")
    targets = [c for c in df.columns if c.startswith("removal_")]
    order = find_run_order(df)
    rows, importance_rows = [], []
    for target in targets:
        d = df[FACTORS + [target]].dropna()
        if len(d) < 10:
            print(f"Skipping {target}: only {len(d)} complete runs")
            continue
        X, y = d[FACTORS].to_numpy(float), d[target].to_numpy(float)
        metal = target.replace("removal_", "")
        preds = {}
        for name, model in make_models().items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                p = cross_val_predict(model, X, y, cv=LeaveOneOut())
            hit_limit = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            preds[name] = p
            row = {"metal": metal, "model": name, "n_runs": len(y),
                   "Q2_LOO": round(q2(y, p), 3),
                   "RMSE_LOO_%": round(float(np.sqrt(np.mean((y - p) ** 2))), 2),
                   "GP_hit_limit": hit_limit if name == "Gaussian process" else ""}
            if name == "Quadratic RSM":
                fitted = model.fit(X, y).predict(X)
                row["R2_in_sample"] = round(1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2), 3)
            rows.append(row)

        # Comparison: how much does the ORDER runs were done in explain, on its own?
        if order:
            p = cross_val_predict(LinearRegression(), df.loc[d.index, [order]].to_numpy(float), y, cv=LeaveOneOut())
            rows.append({"metal": metal, "model": "Run order only (linear)", "n_runs": len(y),
                         "Q2_LOO": round(q2(y, p), 3),
                         "RMSE_LOO_%": round(float(np.sqrt(np.mean((y - p) ** 2))), 2), "GP_hit_limit": ""})

        # Which factors matter? GP length scales (shorter = more influential) and RF importances
        gp = make_models()["Gaussian process"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(X, y)
        ls = gp[-1].kernel_.k1.k2.length_scale
        rf = make_models()["Random forest"].fit(X, y)
        for f, l, imp in zip(FACTORS, np.atleast_1d(ls), rf.feature_importances_):
            importance_rows.append({"metal": metal, "factor": LABELS[f],
                                    "GP_length_scale": round(float(l), 3),
                                    "RF_importance": round(float(imp), 3)})

        # Parity plot: LOO prediction vs measured
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        lo, hi = y.min() - 5, min(y.max() + 5, 105)
        for ax, name in zip(axes, ["Quadratic RSM", "Gaussian process", "Random forest"]):
            ax.scatter(y, preds[name], s=28, color="#1f5fa8", alpha=0.8)
            ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="--")
            ax.set_title(f"{name}\nQ² = {q2(y, preds[name]):.2f}", fontsize=10)
            ax.set_xlabel("Measured removal (%)")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        axes[0].set_ylabel("Predicted for held-out run (%)")
        fig.suptitle(f"{metal}: leave-one-out predictions", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"parity_{metal}.png"), dpi=150)
        plt.close(fig)

    if not rows:
        raise SystemExit("No metal had enough complete runs to model. Is this the empty template file?")
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)
    pd.DataFrame(importance_rows).to_csv(os.path.join(out, "factor_importance.csv"), index=False)

    # Q2 comparison chart across metals
    pivot = summary.pivot(index="metal", columns="model", values="Q2_LOO")
    palette = {"Quadratic RSM": "#9aa5b1", "Gaussian process": "#1f5fa8", "Random forest": "#3a9d5d",
               "Run order only (linear)": "#d1495b"}
    cols = [m for m in palette if m in pivot]
    ax = pivot[cols].plot.bar(figsize=(9, 4), color=[palette[c] for c in cols], width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Q² (leave-one-out)")
    ax.set_xlabel("")
    ax.set_title("Predictive accuracy on held-out runs (higher is better; below 0 = worse than the mean)")
    ax.legend(frameon=False, fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "q2_comparison.png"), dpi=150)
    plt.close()

    diag = diagnostics(df, out)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(summary.drop(columns=["GP_hit_limit"]).to_string(index=False))
        print("\nREPLICATE NOISE AND RUN-ORDER DRIFT")
        print(diag.to_string(index=False))
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="CSV with columns run, castor_g, peel_g, time_min, temp_C, removal_<metal>...")
    ap.add_argument("--demo", action="store_true", help="run on SYNTHETIC data to test the code")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.demo:
        print("DEMO MODE: synthetic data, not thesis results.\n")
        evaluate(demo_data(), a.out + "_DEMO")
    elif a.data:
        evaluate(pd.read_csv(a.data), a.out)
    else:
        ap.error("give --data path/to/your.csv or --demo")
