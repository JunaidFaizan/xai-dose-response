"""Statistics, paper tables and figures from evaluate_explanations.py results.

Usage:
    python analyze_results.py

Reads results/metrics/<dataset>/seed*_{faithfulness,sanity}.csv and
checkpoint/*_metrics.json, and writes results/analysis/:

    accuracy.csv             test accuracy per dataset (mean and std over seeds)
    <dataset>_summary.csv    per-method scores (mean and std over seeds);
                             also shown in the dashboard
    significance.csv         paired Wilcoxon tests between methods, Holm-corrected
    metric_agreement.csv     Kendall tau between the method rankings of each pair
                             of criteria, averaged over datasets and seeds
    gt_agreement.csv         decoy datasets: tau between each metric's ranking and
                             the ground-truth (localization) ranking
    gt_instance.csv          decoy datasets: per-image Spearman correlation between
                             each metric and ground-truth localization
    seed_stability.csv       tau between the rankings from different seeds
    method_ranks.png, metric_agreement.png, gt_agreement.png
    summary.md               the main tables in Markdown
"""

import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon

from xai.data import DATASETS, is_decoy
from xai.explain import METHODS
from xai.metrics import FAITHFULNESS_METRICS, RANDOMIZATION_ORDER
from xai.paths import CHECKPOINT_DIR, RESULTS_DIR

METRICS_DIR = RESULTS_DIR / "metrics"
OUT = RESULTS_DIR / "analysis"
FULLY_RANDOM = RANDOMIZATION_ORDER[-1]
METHOD_TITLES = {name: title for name, (title, _) in METHODS.items()}
METHOD_TITLES["random"] = "Random (baseline)"

# criterion: (title, column, higher_is_better). Sanity uses |correlation|.
CRITERIA = {name: (title, name, higher) for name, (title, _, _, higher) in FAITHFULNESS_METRICS.items()}
CRITERIA["sanity"] = ("Sanity (object px)", "sanity", False)
GROUND_TRUTH = {"mass_accuracy": ("GT mass accuracy", "mass_accuracy", True),
                "rank_accuracy": ("GT rank accuracy", "rank_accuracy", True)}


def load_results():
    faith, sanity = [], []
    for dataset in DATASETS:
        folder = METRICS_DIR / dataset
        faith += [pd.read_csv(p) for p in sorted(folder.glob("seed*_faithfulness.csv"))]
        sanity += [pd.read_csv(p) for p in sorted(folder.glob("seed*_sanity.csv"))]
    if not faith:
        raise SystemExit("No results found. Run evaluate_explanations.py first.")
    faith, sanity = pd.concat(faith), pd.concat(sanity)

    # Attach each image's sanity score (fully randomized model, object pixels)
    # so every criterion is available per (dataset, seed, image, method).
    final = sanity[sanity.stage == FULLY_RANDOM][["dataset", "seed", "index", "method", "corr_object", "corr_all"]]
    final = final.assign(sanity=final.corr_object.abs())
    return faith.merge(final, on=["dataset", "seed", "index", "method"], how="left"), sanity


def goodness(values, higher):
    """Orient scores so that higher always means better."""
    return values if higher else -values


def accuracy_table():
    rows = []
    for dataset in DATASETS:
        files = sorted(CHECKPOINT_DIR.glob(f"{dataset}*_metrics.json"))
        runs = [json.loads(p.read_text()) for p in files]
        runs = [r for r in runs if r.get("dataset") == dataset]
        if not runs:
            continue
        frame = pd.DataFrame(runs)
        row = {"dataset": dataset, "seeds": len(frame)}
        for column in ["test_accuracy", "test_balanced_accuracy", "test_accuracy_no_patch",
                       "test_accuracy_conflicting_patch", "shortcut_reliance"]:
            if column in frame:
                row[f"{column}_mean"] = frame[column].mean()
                row[f"{column}_std"] = frame[column].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows).round(2)


def summary_table(faith, sanity, dataset):
    """Per-method scores for one dataset: mean over seeds of per-seed means, and std over seeds."""
    data = faith[faith.dataset == dataset]
    columns = list(FAITHFULNESS_METRICS) + ["corr_all", "corr_object"]
    if is_decoy(dataset):
        columns += list(GROUND_TRUTH)
    per_seed = data.groupby(["method", "seed"])[columns].mean()
    mean = per_seed.groupby("method").mean()
    std = per_seed.groupby("method").std(ddof=0)

    output_layer = sanity[(sanity.dataset == dataset) & (sanity.stage == "dense4")]
    output_layer = output_layer.groupby(["method", "seed"]).corr_object.mean().groupby("method").mean()

    rows = []
    for method in list(METHODS) + ["random"]:
        if method not in mean.index:
            continue
        row = {"method": METHOD_TITLES[method]}
        for column in columns:
            row[column] = mean.loc[method, column]
            row[f"{column}_std"] = std.loc[method, column]
        row["corr_object_output_layer"] = output_layer.get(method, np.nan)
        rows.append(row)
    table = pd.DataFrame(rows).round(3)
    table.loc[table.method == METHOD_TITLES["random"], ["corr_all", "corr_all_std", "corr_object",
                                                         "corr_object_std", "corr_object_output_layer"]] = np.nan
    return table


def holm(p_values):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def significance(faith):
    """Paired Wilcoxon tests between every pair of methods, per dataset and criterion."""
    rows = []
    names = list(METHODS) + ["random"]
    for dataset in faith.dataset.unique():
        data = faith[faith.dataset == dataset]
        criteria = dict(CRITERIA, **(GROUND_TRUTH if is_decoy(dataset) else {}))
        for criterion, (title, column, higher) in criteria.items():
            wide = data.pivot_table(index=["seed", "index"], columns="method", values=column)
            block = []
            for a, b in itertools.combinations(names, 2):
                if criterion == "sanity" and "random" in (a, b):
                    continue
                if a not in wide or b not in wide:
                    continue
                pair = wide[[a, b]].dropna()
                diff = goodness(pair[a] - pair[b], higher)
                try:
                    p = wilcoxon(pair[a], pair[b]).pvalue if (diff != 0).any() else 1.0
                except ValueError:
                    p = 1.0
                block.append({"dataset": dataset, "criterion": title, "method_a": METHOD_TITLES[a],
                              "method_b": METHOD_TITLES[b], "n": len(pair),
                              "median_advantage_a": float(np.median(diff)),
                              "better": METHOD_TITLES[a] if np.median(diff) > 0 else METHOD_TITLES[b],
                              "p_value": p})
            if block:
                adjusted = holm([r["p_value"] for r in block])
                for r, p_adj in zip(block, adjusted):
                    r["p_holm"] = p_adj
                    r["significant"] = p_adj < 0.05
                rows += block
    return pd.DataFrame(rows)


def method_scores(faith, criteria):
    """Mean score of each method (excluding random) per dataset, seed and criterion."""
    data = faith[faith.method != "random"]
    columns = [column for _, column, _ in criteria.values()]
    return data.groupby(["dataset", "seed", "method"])[columns].mean()


def kendall(a, b):
    tau = kendalltau(a, b).statistic
    return np.nan if np.isnan(tau) else float(tau)


def metric_agreement(faith):
    scores = method_scores(faith, CRITERIA)
    names = list(CRITERIA)
    taus = {pair: [] for pair in itertools.product(names, names)}
    for (_, _), group in scores.groupby(level=["dataset", "seed"]):
        for a, b in itertools.product(names, names):
            ga = goodness(group[CRITERIA[a][1]], CRITERIA[a][2])
            gb = goodness(group[CRITERIA[b][1]], CRITERIA[b][2])
            taus[(a, b)].append(kendall(ga, gb))
    matrix = pd.DataFrame(index=[CRITERIA[n][0] for n in names], columns=[CRITERIA[n][0] for n in names], dtype=float)
    for (a, b), values in taus.items():
        matrix.loc[CRITERIA[a][0], CRITERIA[b][0]] = np.nanmean(values)
    return matrix.round(3)


def gt_agreement(faith):
    """Decoy datasets: does each metric rank methods like the ground truth does?"""
    decoys = [d for d in faith.dataset.unique() if is_decoy(d)]
    data = faith[faith.dataset.isin(decoys)]
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    scores = method_scores(data, dict(CRITERIA, **GROUND_TRUTH))

    ranking_rows = []
    for (dataset, seed), group in scores.groupby(level=["dataset", "seed"]):
        truth = group["mass_accuracy"]
        for name, (title, column, higher) in CRITERIA.items():
            ranking_rows.append({"dataset": dataset, "seed": seed, "metric": title,
                                 "tau_with_ground_truth": kendall(goodness(group[column], higher), truth)})
    ranking = pd.DataFrame(ranking_rows)

    # Per-image: across all (image, method) pairs, does a better metric score
    # go with better localization of the true evidence?
    instance_rows = []
    per_image = data[data.method != "random"]
    for dataset in decoys:
        subset = per_image[per_image.dataset == dataset]
        for name, (title, column, higher) in CRITERIA.items():
            pair = subset[[column, "mass_accuracy"]].dropna()
            rho = spearmanr(goodness(pair[column], higher), pair["mass_accuracy"]).statistic
            instance_rows.append({"dataset": dataset, "metric": title, "spearman_with_ground_truth": rho,
                                  "n": len(pair)})
    return ranking, pd.DataFrame(instance_rows)


def seed_stability(faith):
    """How consistently does each criterion rank the methods across seeds?"""
    scores = method_scores(faith, CRITERIA)
    rows = []
    for dataset, group in scores.groupby(level="dataset"):
        seeds = group.index.get_level_values("seed").unique()
        for name, (title, column, higher) in CRITERIA.items():
            taus = []
            for s1, s2 in itertools.combinations(seeds, 2):
                a = goodness(group.xs(s1, level="seed")[column], higher)
                b = goodness(group.xs(s2, level="seed")[column], higher)
                taus.append(kendall(a, b.reindex(a.index)))
            rows.append({"dataset": dataset, "criterion": title,
                         "mean_tau_between_seeds": np.nanmean(taus) if taus else np.nan})
    return pd.DataFrame(rows).round(3)


def plot_method_ranks(faith, path):
    """Rank of each method (1 = best) under each criterion, per dataset."""
    datasets = [d for d in DATASETS if d in faith.dataset.unique()]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.9), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        criteria = dict(CRITERIA, **({"mass_accuracy": GROUND_TRUTH["mass_accuracy"]} if is_decoy(dataset) else {}))
        scores = method_scores(faith[faith.dataset == dataset], criteria).groupby("method").mean()
        ranks = pd.DataFrame({
            title: goodness(scores[column], higher).rank(ascending=False)
            for title, column, higher in criteria.values()
        }).reindex(list(METHODS))
        ax.imshow(ranks.values, cmap="RdYlGn_r", vmin=1, vmax=len(METHODS))
        for (i, j), value in np.ndenumerate(ranks.values):
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(ranks.columns)), ranks.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(ranks.index)), [METHOD_TITLES[m] for m in ranks.index], fontsize=8)
        ax.set_title(dataset, fontsize=10)
    fig.suptitle("Method rank under each criterion (1 = best, mean score over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_matrix(matrix, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix.values.astype(float), cmap="coolwarm", vmin=-1, vmax=1)
    for (i, j), value in np.ndenumerate(matrix.values):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    fig.colorbar(image, ax=ax, label="Kendall tau")
    ax.set_title("Agreement between evaluation criteria\n(mean Kendall tau over datasets and seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_gt(ranking, instance, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    order = [CRITERIA[n][0] for n in CRITERIA]
    for ax, (frame, column, title) in zip(axes, [
        (ranking, "tau_with_ground_truth", "Method ranking vs ground truth\n(Kendall tau, mean ± std over seeds)"),
        (instance, "spearman_with_ground_truth", "Per-image score vs ground truth\n(Spearman rho)"),
    ]):
        datasets = frame.dataset.unique()
        width = 0.8 / len(datasets)
        for k, dataset in enumerate(datasets):
            stats = frame[frame.dataset == dataset].groupby("metric")[column].agg(["mean", "std"]).reindex(order)
            ax.bar(np.arange(len(order)) + k * width, stats["mean"], width,
                   yerr=stats["std"].fillna(0), capsize=3, label=dataset)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(np.arange(len(order)) + width * (len(datasets) - 1) / 2, order, rotation=30, ha="right")
        ax.set_ylim(-1, 1)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markdown(frame, floatfmt=3):
    """Simple Markdown table without extra dependencies."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else f"{v:.{floatfmt}f}"
        return str(v)
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    faith, sanity = load_results()
    datasets = [d for d in DATASETS if d in faith.dataset.unique()]
    lines = ["# Results summary", "", "Generated by `analyze_results.py`. Means over seeds; ± is the std over seeds.", ""]

    acc = accuracy_table()
    acc.to_csv(OUT / "accuracy.csv", index=False)
    lines += ["## Model accuracy (%)", "", markdown(acc, 2), ""]

    for dataset in datasets:
        table = summary_table(faith, sanity, dataset)
        table.to_csv(OUT / f"{dataset}_summary.csv", index=False)
        cols = ["method"] + list(FAITHFULNESS_METRICS) + ["corr_object"] + \
               (list(GROUND_TRUTH) if is_decoy(dataset) else [])
        lines += [f"## {dataset}", "", markdown(table[cols]), ""]

    sig = significance(faith)
    sig.to_csv(OUT / "significance.csv", index=False)
    share = sig.groupby(["dataset", "criterion"]).significant.mean().mul(100).round(0).reset_index()
    lines += ["## Share of method pairs with a significant difference (Wilcoxon, Holm, p < 0.05)", "",
              markdown(share.rename(columns={"significant": "% significant"}), 0), ""]

    agreement = metric_agreement(faith)
    agreement.to_csv(OUT / "metric_agreement.csv")
    plot_matrix(agreement, OUT / "metric_agreement.png")
    lines += ["## Agreement between criteria (mean Kendall tau)", "",
              markdown(agreement.reset_index().rename(columns={"index": "criterion"})), ""]

    ranking, instance = gt_agreement(faith)
    if not ranking.empty:
        ranking.to_csv(OUT / "gt_agreement.csv", index=False)
        instance.to_csv(OUT / "gt_instance.csv", index=False)
        plot_gt(ranking, instance, OUT / "gt_agreement.png")
        gt_mean = ranking.groupby(["dataset", "metric"]).tau_with_ground_truth.agg(["mean", "std"]).reset_index()
        lines += ["## Decoy datasets: metric ranking vs ground truth (Kendall tau)", "", markdown(gt_mean), "",
                  "## Decoy datasets: per-image agreement with ground truth (Spearman rho)", "",
                  markdown(instance), ""]

    stability = seed_stability(faith)
    stability.to_csv(OUT / "seed_stability.csv", index=False)
    lines += ["## Ranking stability across seeds (mean Kendall tau between seeds)", "", markdown(stability), ""]

    plot_method_ranks(faith, OUT / "method_ranks.png")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved analysis to {OUT}")


if __name__ == "__main__":
    main()
