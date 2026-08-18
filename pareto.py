import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Pareto plotting for benchmark CSV")
    parser.add_argument("--csv-file", default="Pareto.csv")
    parser.add_argument("--output-dir", default="Pareto")
    parser.add_argument("--mode", choices=["methods", "topk", "cluster"], default="methods")
    parser.add_argument("--baseline", default="Baseline")
    parser.add_argument(
        "--method",
        nargs="+",
        default=None,
        help="Required for --mode topk; supports comma-separated values",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets; supports comma-separated values or space-separated tokens",
    )
    parser.add_argument(
        "--methods-topk",
        default="15",
        help="In methods mode, filter to this steer.top_k value, or use 'all'",
    )
    parser.add_argument(
        "--topk-values",
        nargs="+",
        default=None,
        help="In topk mode, supports comma-separated top_k values; default uses all for method",
    )
    parser.add_argument(
        "--cluster-width",
        type=float,
        default=0.5,
        help="In cluster mode, perplexity bin width used to group near values",
    )
    parser.add_argument(
        "--cluster-topk",
        nargs="+",
        default=None,
        help="In cluster mode, optional top_k filter; supports comma-separated values",
    )
    parser.add_argument("--clip-sigma", type=float, default=None)
    parser.add_argument("--acc-bounds", nargs=2, type=float, default=[80.0, 100.0])
    parser.add_argument("--neg-ppl-bounds", nargs=2, type=float, default=[-10, -1])
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    return parser.parse_args()


def is_pareto_optimal(points):
    pts = np.array(points)
    result = []
    for i, p in enumerate(pts):
        dominated = any(
            all(pts[j] >= p) and any(pts[j] > p)
            for j in range(len(pts))
            if j != i
        )
        result.append(not dominated)
    return result


def clip_bounds(data, col_x, col_y, sigma, x_bounds=None, y_bounds=None):
    if x_bounds is not None:
        xlo, xhi = x_bounds
    else:
        xlo, xhi = data[col_x].min(), data[col_x].max()

    if y_bounds is not None:
        ylo, yhi = y_bounds
    else:
        ylo, yhi = data[col_y].min(), data[col_y].max()

    if sigma is None:
        return xlo, xhi, ylo, yhi

    mx, sx = data[col_x].mean(), data[col_x].std()
    my, sy = data[col_y].mean(), data[col_y].std()
    if x_bounds is None:
        xlo, xhi = mx - sigma * sx, mx + sigma * sx
    if y_bounds is None:
        ylo, yhi = my - sigma * sy, my + sigma * sy
    return xlo, xhi, ylo, yhi


def draw_baseline(ax, sub, baseline):
    bsub = sub[sub["method"] == baseline]
    if not bsub.empty:
        b_acc = bsub["accuracy"].mean()
        b_ppl = bsub["neg_ppl"].mean()
        ax.axvline(b_acc, color="gray", ls="--", lw=1.2, alpha=0.55)
        ax.axhline(b_ppl, color="gray", ls="--", lw=1.2, alpha=0.55)


def get_datasets(df, dataset_arg):
    if dataset_arg is None:
        return list(df["dataset"].dropna().unique())
    return parse_multi_values(dataset_arg)


def parse_multi_values(value):
    if value is None:
        return []
    if isinstance(value, str):
        raw = value
    else:
        raw = " ".join(str(x) for x in value)
    parts = []
    for token in raw.split(","):
        s = token.strip()
        if s:
            parts.extend([x for x in s.split() if x])
    # Preserve order and drop duplicates.
    seen = set()
    result = []
    for x in parts:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def is_sae_method(method_name):
    method = str(method_name).upper()
    if "SAE" in method:
        return True
    return method in {"SAS", "SPARE", "SRPS", "CORRSTEER", "SSV", "SRE"}

def format_setup_label(row):
    c = row.get("coefficient", None)
    k = row.get("steer.top_k", None)

    parts = []

    if pd.notna(c):
        parts.append(f"c={c}")

    if pd.notna(k):
        # handle cases like "top3", "top_5", "5", etc.
        if isinstance(k, str):
            k_clean = k.replace("top", "").replace("_", "")
            if k_clean.isdigit():
                parts.append(f"k={int(k_clean)}")
            else:
                parts.append(f"k={k}")  # fallback
        else:
            parts.append(f"k={int(k)}")

    return " ".join(parts)

def build_cluster_best_per_method(sub, cluster_width):
    """Cluster by perplexity and keep highest-accuracy setup per method/cluster."""
    clustered = sub.copy()
    clustered["ppl_cluster"] = np.floor(clustered["perplexity"] / cluster_width).astype(int)
    clustered["cluster_center"] = (clustered["ppl_cluster"] + 0.5) * cluster_width

    return (
        clustered.sort_values(
            ["method", "ppl_cluster", "accuracy", "neg_ppl"],
            ascending=[True, True, False, False],
        )
        .groupby(["method", "ppl_cluster"], as_index=False)
        .first()
        .sort_values(["method", "cluster_center"])
        .reset_index(drop=True)
    )

def select_best_per_cluster_with_tiebreak(rows):
    """
    For each cluster:
    1. keep max accuracy
    2. if tie → keep lowest perplexity
    3. DO NOT collapse by method
    """
    if rows.empty:
        return rows.copy()

    return (
        rows.sort_values(
            ["ppl_cluster", "accuracy", "perplexity"],
            ascending=[True, False, True],  # key change
        )
        .groupby(["ppl_cluster"], as_index=False)
        .first()
        .sort_values(["cluster_center"])
        .reset_index(drop=True)
    )

def build_cluster_details_csv(best_rows, plotted_rows):
    """Build detail table containing all candidates and whether each point is plotted."""
    details = best_rows.copy()
    details["setup"] = details.apply(format_setup_label, axis=1)

    key_cols = [
        "method",
        "ppl_cluster",
        "accuracy",
        "perplexity",
        "coefficient",
        "steer.top_k",
    ]
    picked = plotted_rows[key_cols].copy()
    picked["selected_for_plot"] = True
    if "point_id" in plotted_rows.columns:
        picked["point_id"] = plotted_rows["point_id"].values
    else:
        picked["point_id"] = np.nan

    details = details.merge(picked.drop_duplicates(), on=key_cols, how="left")
    details["selected_for_plot"] = details["selected_for_plot"].eq(True)
    details["point_id"] = details["point_id"].astype("Int64")

    return details[
        [
            "point_id",
            "selected_for_plot",
            "setup",
            "method",
            "coefficient",
            "steer.top_k",
            "accuracy",
            "perplexity",
            "ppl_cluster",
            "cluster_center",
        ]
    ].sort_values(["selected_for_plot", "accuracy", "perplexity"], ascending=[False, False, True])


def methods_mode(df, args):
    datasets = get_datasets(df, args.datasets)
    methods = [m for m in df["method"].unique() if m != args.baseline]
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

    for dataset in datasets:
        sub = df[df["dataset"] == dataset].copy()
        if sub.empty:
            continue

        if args.methods_topk.lower() != "all":
            sub = sub[(sub["method"] == args.baseline) | (sub["steer.top_k"] == args.methods_topk)]

        fig, ax = plt.subplots(figsize=(7, 5))

        non_base = sub[sub["method"] != args.baseline]
        if non_base.empty:
            plt.close(fig)
            continue

        x_bounds = tuple(args.acc_bounds) if args.acc_bounds is not None else None
        y_bounds = tuple(args.neg_ppl_bounds) if args.neg_ppl_bounds is not None else None
        xlo, xhi, ylo, yhi = clip_bounds(
            non_base,
            "accuracy",
            "neg_ppl",
            args.clip_sigma,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )

        pad_x = max((xhi - xlo) * 0.12, 1e-3)
        pad_y = max((yhi - ylo) * 0.12, 1e-3)
        ax.set_xlim(xlo - pad_x, xhi + pad_x)
        ax.set_ylim(ylo - pad_y, yhi + pad_y)

        all_visible = []
        visible_methods = []

        for method in methods:
            msub = sub[sub["method"] == method].sort_values("coefficient")
            pts = msub[["accuracy", "neg_ppl", "coefficient"]].dropna().values
            if len(pts) == 0:
                continue

            col = color_map[method]
            visible = [(a, n, c) for a, n, c in pts if xlo <= a <= xhi and ylo <= n <= yhi]
            if not visible:
                continue

            visible_methods.append(method)
            all_visible.extend([(a, n) for a, n, _ in visible])
            ax.scatter([p[0] for p in visible], [p[1] for p in visible], color=col, s=55, zorder=4)

            for i in range(len(visible) - 1):
                x1, y1, _ = visible[i]
                x2, y2, _ = visible[i + 1]
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>", color=col, lw=1.4, mutation_scale=11, alpha=0.75
                    ),
                )

        all_visible = list(set(all_visible))
        if len(all_visible) >= 2:
            flags = is_pareto_optimal(all_visible)
            front = sorted([p for p, f in zip(all_visible, flags) if f], key=lambda p: p[0])
            if len(front) >= 2:
                px, py = zip(*front)
                ax.plot(px, py, color="#534AB7", lw=2, zorder=3, alpha=0.85)

        draw_baseline(ax, sub, args.baseline)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("-Perplexity (higher = better)")
        ax.set_title(f"Pareto frontier - {dataset}", fontweight="normal")

        handles = [mpatches.Patch(color=color_map[m], label=m) for m in visible_methods]
        handles += [
            plt.Line2D([0], [0], color="#534AB7", lw=2, label="Pareto frontier"),
            plt.Line2D([0], [0], color="gray", ls="--", lw=1.2, label="Baseline"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="lower right")

        plt.tight_layout()
        out = os.path.join(args.output_dir, f"pareto_{dataset}.png")
        plt.savefig(out, dpi=150)
        if args.show:
            plt.show()
        plt.close(fig)
        print(f"Saved {out}")


def topk_mode(df, args):
    method_values = parse_multi_values(args.method)
    if not method_values:
        raise ValueError("--method is required when --mode topk")

    datasets = get_datasets(df, args.datasets)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    topk_arg_values = parse_multi_values(args.topk_values)
    any_saved = False

    for method in method_values:
        for dataset in datasets:
            sub = df[(df["dataset"] == dataset) & (df["method"] == method)].copy()
            if sub.empty:
                continue

            topk_values = (
                topk_arg_values
                if topk_arg_values
                else sorted(sub["steer.top_k"].dropna().astype(str).unique())
            )
            sub = sub[sub["steer.top_k"].astype(str).isin(topk_values)]
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(7, 5))
            non_base = sub.copy()
            x_bounds = tuple(args.acc_bounds) if args.acc_bounds is not None else None
            y_bounds = tuple(args.neg_ppl_bounds) if args.neg_ppl_bounds is not None else None
            xlo, xhi, ylo, yhi = clip_bounds(
                non_base,
                "accuracy",
                "neg_ppl",
                args.clip_sigma,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
            )
            pad_x = max((xhi - xlo) * 0.12, 1e-3)
            pad_y = max((yhi - ylo) * 0.12, 1e-3)
            ax.set_xlim(xlo - pad_x, xhi + pad_x)
            ax.set_ylim(ylo - pad_y, yhi + pad_y)

            color_map = {k: palette[i % len(palette)] for i, k in enumerate(topk_values)}
            all_visible = []
            visible_topk = []

            for topk in topk_values:
                ksub = sub[sub["steer.top_k"].astype(str) == str(topk)].sort_values("coefficient")
                pts = ksub[["accuracy", "neg_ppl", "coefficient"]].dropna().values
                if len(pts) == 0:
                    continue

                col = color_map[topk]
                visible = [(a, n, c) for a, n, c in pts if xlo <= a <= xhi and ylo <= n <= yhi]
                if not visible:
                    continue

                visible_topk.append(topk)
                all_visible.extend([(a, n) for a, n, _ in visible])
                ax.scatter([p[0] for p in visible], [p[1] for p in visible], color=col, s=55, zorder=4)

                for i in range(len(visible) - 1):
                    x1, y1, _ = visible[i]
                    x2, y2, _ = visible[i + 1]
                    ax.annotate(
                        "",
                        xy=(x2, y2),
                        xytext=(x1, y1),
                        arrowprops=dict(
                            arrowstyle="-|>", color=col, lw=1.4, mutation_scale=11, alpha=0.75
                        ),
                    )

            all_visible = list(set(all_visible))
            if len(all_visible) >= 2:
                flags = is_pareto_optimal(all_visible)
                front = sorted([p for p, f in zip(all_visible, flags) if f], key=lambda p: p[0])
                if len(front) >= 2:
                    px, py = zip(*front)
                    ax.plot(px, py, color="#534AB7", lw=2, zorder=3, alpha=0.85)

            base_sub = df[(df["dataset"] == dataset) & (df["method"] == args.baseline)]
            draw_baseline(ax, pd.concat([base_sub, sub], axis=0), args.baseline)

            ax.set_xlabel("Accuracy")
            ax.set_ylabel("-Perplexity (higher = better)")
            ax.set_title(f"Pareto by top_k - {method} - {dataset}", fontweight="normal")

            handles = [mpatches.Patch(color=color_map[k], label=f"top_k={k}") for k in visible_topk]
            handles += [
                plt.Line2D([0], [0], color="#534AB7", lw=2, label="Pareto frontier"),
                plt.Line2D([0], [0], color="gray", ls="--", lw=1.2, label="Baseline"),
            ]
            ax.legend(handles=handles, fontsize=8, loc="lower right")

            plt.tight_layout()
            out = os.path.join(args.output_dir, f"pareto_{method}_{dataset}_topk.png")
            plt.savefig(out, dpi=150)
            if args.show:
                plt.show()
            plt.close(fig)
            any_saved = True
            print(f"Saved {out}")

    if not any_saved:
        print("No plots generated. Check --method/--datasets values and available CSV rows.")

def cluster_mode(df, args):
    if args.cluster_width <= 0:
        raise ValueError("--cluster-width must be > 0")

    datasets = get_datasets(df, args.datasets)
    method_values = parse_multi_values(args.method)
    topk_filter = parse_multi_values(args.cluster_topk)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    any_saved = False

    for dataset in datasets:
        sub = df[(df["dataset"] == dataset) & (df["method"] != args.baseline)].copy()
        if method_values:
            sub = sub[sub["method"].isin(method_values)]
        if topk_filter:
            sub = sub[sub["steer.top_k"].astype(str).isin(topk_filter)]

        sub = sub.dropna(subset=["accuracy", "perplexity", "neg_ppl"])
        if args.acc_bounds is not None:
            sub = sub[
                (sub["accuracy"] >= args.acc_bounds[0])
                & (sub["accuracy"] <= args.acc_bounds[1])
            ]
        if args.neg_ppl_bounds is not None:
            sub = sub[
                (sub["neg_ppl"] >= args.neg_ppl_bounds[0])
                & (sub["neg_ppl"] <= args.neg_ppl_bounds[1])
            ]

        if sub.empty:
            continue

        best = build_cluster_best_per_method(sub, args.cluster_width)

        # Keep all per-method / per-cluster winners.
        # Do NOT collapse again across clusters or methods.
        plotted = best.copy().reset_index(drop=True)

        if plotted.empty:
            continue

        plotted["point_id"] = np.arange(1, len(plotted) + 1)

        methods = list(plotted["method"].dropna().unique())
        color_map = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

        fig, ax = plt.subplots(figsize=(9, 5.6))

        # Highlight each perplexity cluster region so cross-method points are compared locally.
        cluster_bins = sorted(best["ppl_cluster"].unique())
        for i, cid in enumerate(cluster_bins):
            left = cid * args.cluster_width
            right = (cid + 1) * args.cluster_width
            alpha = 0.10 if i % 2 == 0 else 0.05
            ax.axvspan(left, right, color="#AAB2BF", alpha=alpha, zorder=0)

        for method in methods:
            msub = plotted[plotted["method"] == method].sort_values("cluster_center")
            if msub.empty:
                continue

            ax.scatter(
                msub["cluster_center"],
                msub["accuracy"],
                marker="o",
                s=60,
                color=color_map[method],
                label=method,
                alpha=0.92,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )

            for i, (_, row) in enumerate(msub.iterrows()):
                ax.annotate(
                    format_setup_label(row),
                    (row["cluster_center"], row["accuracy"]),
                    textcoords="offset points",
                    xytext=(0, 6 + (i % 4) * 4),   # stagger vertically
                    ha="center",
                    va="bottom",
                    fontsize=6.3,
                    color=color_map[method],
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        fc="white",
                        ec=color_map[method],
                        lw=0.6,
                        alpha=0.9,
                    ),
                    zorder=4,
                )

        ax.set_xlabel("Perplexity clusters")
        ax.set_ylabel("Best accuracy in cluster")
        ax.set_title(
            f"Cross-method comparison by perplexity clusters - {dataset}",
            fontweight="normal",
        )
        ax.grid(alpha=0.25, ls="--", lw=0.6)

        method_legend = ax.legend(
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            title="Methods",
            title_fontsize=8,
        )
        ax.add_artist(method_legend)

        plt.tight_layout(rect=[0, 0, 0.82, 1])
        out = os.path.join(args.output_dir, f"pareto_cluster_{dataset}.png")
        plt.savefig(out, dpi=150)
        if args.show:
            plt.show()
        plt.close(fig)

        # Save all per-method cluster winners and indicate plotted representatives.
        mapping = build_cluster_details_csv(best, plotted)
        map_out = os.path.join(args.output_dir, f"pareto_cluster_{dataset}_points.csv")
        mapping.to_csv(map_out, index=False)

        any_saved = True
        print(f"Saved {out}")
        print(f"Saved {map_out}")

    if not any_saved:
        print("No cluster plots generated. Check dataset/method/top_k filters and bounds.")
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_file)
    for col in ["coefficient", "accuracy", "perplexity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "steer.top_k" not in df.columns:
        df["steer.top_k"] = ""
    df["steer.top_k"] = df["steer.top_k"].fillna("").astype(str)
    df.loc[(df["method"] != args.baseline) & (df["steer.top_k"] == ""), "steer.top_k"] = "15"
    df["neg_ppl"] = -df["perplexity"]

    if args.mode == "methods":
        methods_mode(df, args)
    elif args.mode == "topk":
        topk_mode(df, args)
    else:
        cluster_mode(df, args)


if __name__ == "__main__":
    main()