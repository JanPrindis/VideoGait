import argparse
import os
import sys

# Path hack
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Remove the script directory from sys.path to prevent module shadowing
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import yaml

from utils.jsonSerializer import AnnotationSerializer
from gaitStructs import GaitEventType

from benchmark.utils.data_manager import get_benchmark_files
from benchmark.engine.matcher import match_events_greedy
from benchmark.wrappers.nn_wrapper import NeuralNetWrapper
from benchmark.wrappers.heuristic_wrapper import HeuristicWrapper


# --- PLOTTING STYLE CONFIGURATION ---
def set_publication_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 1.5,
        'figure.autolayout': False
    })


def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true', help="Enable verbose output for the first file")
    args = parser.parse_args()

    # Load Config
    print(f"--- Loading Config: {args.config} ---")
    with open(os.path.join(PROJECT_ROOT, args.config)) as f:
        cfg = yaml.safe_load(f)

    experiment_name = cfg.get('experiment_name', 'Benchmark')

    # Get Data (Files + resolved FPS)
    test_files, global_fps = get_benchmark_files(cfg)

    # Setup Wrapper
    method = cfg['event_detector']['method']
    if method == "NeuralNet":
        wrapper = NeuralNetWrapper(cfg['preprocessing'], cfg['event_detector']['neural_net'])

    elif method == "Heuristic":
        wrapper = HeuristicWrapper(cfg['preprocessing'], cfg['event_detector']['heuristic'])

    else:
        raise ValueError(f"Unknown method: {method}")

    # Dual tolerance
    # Detections under loose tolerance are shown in the barplots and histograms
    # Detections under strict tolerance are used to calculate F1 score and associated metrics
    strict_tolerance_ms = cfg['evaluation'].get('strict_tolerance_ms', 50)
    loose_tolerance_ms = cfg['evaluation'].get('loose_tolerance_ms', 150)

    print(f"\n--- Starting Benchmark: {experiment_name} ---")
    print(f"[Eval] Loose Tolerance (Graphs): {loose_tolerance_ms} ms")
    print(f"[Eval] Strict Tolerance (Stats): {strict_tolerance_ms} ms")

    # Evaluation Loop Containers
    raw_matches = []

    total_counts = {
        "left": {et.value: {"Total_GT": 0, "Total_Pred": 0} for et in
                 [GaitEventType.HEEL_STRIKE, GaitEventType.TOE_OFF]},
        "right": {et.value: {"Total_GT": 0, "Total_Pred": 0} for et in
                  [GaitEventType.HEEL_STRIKE, GaitEventType.TOE_OFF]}
    }

    debug_active = args.debug

    for i, item in enumerate(test_files):
        print(f"[{i + 1}/{len(test_files)}] {item['name']}...", end="\r")

        # Predict
        pred_result = wrapper.predict(item['kp'])
        if not pred_result: continue

        current_fps = pred_result.get('framerate', global_fps)

        # Tolerance MS -> FPS conversion
        loose_tolerance_frames = int((loose_tolerance_ms / 1000.0) * current_fps)
        valid_ranges = pred_result['global_ranges']

        # Load Ground Truth
        gt_data = AnnotationSerializer.load(item['ann'])

        for side in ['left', 'right']:
            gt_events_list = gt_data['annotations'][side]

            for et in [GaitEventType.HEEL_STRIKE, GaitEventType.TOE_OFF]:
                predictions = [e.frame for e in pred_result['events'][side] if e.event_type == et]
                gt_raw = [e.frame for e in gt_events_list if e.event_type == et]

                # Exclude borders logic
                gt_valid = []
                for f in gt_raw:
                    if any(start <= f <= end for start, end in valid_ranges):
                        gt_valid.append(f)

                total_counts[side][et.value]["Total_GT"] += len(gt_valid)
                total_counts[side][et.value]["Total_Pred"] += len(predictions)

                should_print = debug_active and (i == 0)

                # Matching with loose tolerance (for plot data)
                errors, misses, extras = match_events_greedy(
                    gt_valid,
                    predictions,
                    loose_tolerance_frames,
                    verbose=should_print,
                    label=f"{item['name']} | {side} | {et.name}"
                )

                # Save loose matches
                for err_frames in errors:
                    err_ms = err_frames * (1000.0 / current_fps)
                    raw_matches.append({
                        "File": item['name'],
                        "FPS": current_fps,
                        "Side": side,
                        "Type": et.value,
                        "ErrorFrames": err_frames,
                        "AbsErrorFrames": abs(err_frames),
                        "ErrorMs": err_ms,
                        "AbsErrorMs": abs(err_ms)
                    })

        if i == 0 and debug_active:
            print(f"\n[Info] Debug print disabled for remaining {len(test_files) - 1} files...\n")
            debug_active = False

    print("\n\n--- Processing Results & Generating Plots ---")
    out_dir = os.path.join(PROJECT_ROOT, cfg['output_dir'])
    os.makedirs(out_dir, exist_ok=True)
    set_publication_style()

    # --- PROCESS DATA ---
    df_raw = pd.DataFrame(raw_matches)  # Toto obsahuje LOOSE matches
    if df_raw.empty:
        print("No matches found! Check tolerance or data.")
        return

    # Save Loose Matches
    df_raw.to_csv(os.path.join(out_dir, "raw_matches_loose.csv"), index=False)

    # Stats Summary - Loose matches
    desc_frames = df_raw.groupby("Type")["ErrorFrames"].describe()
    desc_ms = df_raw.groupby("Type")["ErrorMs"].describe()
    mae_frames = df_raw.groupby("Type")["AbsErrorFrames"].mean()
    mae_ms = df_raw.groupby("Type")["AbsErrorMs"].mean()

    stats_summary = pd.concat([
        desc_frames.add_suffix('_Frames'),
        mae_frames.rename("MAE_Frames"),
        desc_ms.add_suffix('_Ms'),
        mae_ms.rename("MAE_Ms")
    ], axis=1)
    stats_summary.to_csv(os.path.join(out_dir, "stats_summary_loose.csv"))

    # --- RELIABILITY REPORT ---
    print(f"[Metrics] Calculating scores with STRICT tolerance ({strict_tolerance_ms} ms)...")

    # Filter out data not matching strict tolerance
    df_strict_matches = df_raw[df_raw['AbsErrorMs'] <= strict_tolerance_ms]

    rel_rows = []
    for side in ['left', 'right']:
        for et_val in [GaitEventType.HEEL_STRIKE.value, GaitEventType.TOE_OFF.value]:
            tot_gt = total_counts[side][et_val]["Total_GT"]
            tot_pred = total_counts[side][et_val]["Total_Pred"]

            # TP
            tp = len(df_strict_matches[
                         (df_strict_matches['Side'] == side) &
                         (df_strict_matches['Type'] == et_val)
                         ])

            # FN = Predictions that are not found in strict tolerance (Total GT - TP_strict)
            fn = tot_gt - tp
            # FP = Predictions that are not TP_strict (Total Pred - TP_strict)
            fp = tot_pred - tp

            fn = max(0, fn)
            fp = max(0, fp)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            rel_rows.append({
                "Side": side, "Type": et_val,
                "TP": tp, "FN": fn, "FP": fp,
                "Total_GT": tot_gt,
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1_Score": round(f1, 4)
            })

    df_rel = pd.DataFrame(rel_rows)
    df_rel.to_csv(os.path.join(out_dir, "reliability_report_strict.csv"), index=False)

    print("Strict Metrics:")
    print(df_rel[['Side', 'Type', 'Precision', 'Recall', 'F1_Score']])

    # --- PLOTS ---
    # Reliability Barplot - STRICT
    df_rel_agg = df_rel.groupby("Type")[["TP", "FN", "FP"]].sum().reset_index()
    df_rel_melted = df_rel_agg.melt(id_vars="Type", value_vars=["TP", "FN", "FP"],
                                    var_name="Category", value_name="Count")

    plt.figure(figsize=(5, 5))
    sns.barplot(
        data=df_rel_melted, x="Type", y="Count", hue="Category",
        palette={"TP": "#2ecc71", "FN": "#e74c3c", "FP": "#f39c12"},
        edgecolor="black", linewidth=0.5, width=0.7
    )
    plt.title(f"Reliability ({strict_tolerance_ms}ms error threshold)", fontweight='bold')
    plt.ylabel("Count")
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title=None, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reliability_barplot_strict.png"))

    # Accuracy Boxplot (Frames) - LOOSE
    plt.figure(figsize=(4, 5))
    ax = sns.boxplot(
        data=df_raw, x="Type", y="ErrorFrames",
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5, "markerfacecolor": "gray"},
        width=0.5,
        boxprops=dict(facecolor='#3498db', alpha=0.8)
    )
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.title(f"Accuracy (Frames) - Loose {loose_tolerance_ms}ms", fontweight='bold')
    plt.ylabel("Error (Frames)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_boxplot_frames.png"))

    # Accuracy Boxplot (ms) - LOOSE
    plt.figure(figsize=(4, 5))
    sns.boxplot(
        data=df_raw, x="Type", y="ErrorMs",
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5, "markerfacecolor": "gray"},
        width=0.5,
        boxprops=dict(facecolor='#9b59b6', alpha=0.8)
    )
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.title(f"Accuracy (Time) - Loose {loose_tolerance_ms}ms", fontweight='bold')
    plt.ylabel("Error (ms)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_boxplot_ms.png"))

    # Error Distribution (Frames) - LOOSE
    df_hist = df_raw.copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    events = [GaitEventType.HEEL_STRIKE.value, GaitEventType.TOE_OFF.value]
    titles = ["Heel Strike", "Toe Off"]
    colors = ["#3498db", "#e74c3c"]

    for ax, event, title, color in zip(axes, events, titles, colors):
        subset = df_hist[df_hist["Type"] == event]

        if subset.empty:
            continue

        # Determine Range
        vals = subset['ErrorFrames'].astype(int)
        min_val = vals.min()
        max_val = vals.max()

        plot_min = min(min_val, 0) - 2
        plot_max = max(max_val, 0) + 2
        full_range = list(range(plot_min, plot_max + 1))

        # Fill missing values
        counts = vals.value_counts().reindex(full_range, fill_value=0)
        dist_data = counts.reset_index()
        dist_data.columns = ['ErrorFrames', 'Count']

        # Plot
        sns.barplot(
            data=dist_data, x="ErrorFrames", y="Count",
            ax=ax, color=color, edgecolor="black", alpha=0.8
        )

        # Ticks
        tick_idxs = range(len(full_range))
        tick_labels = full_range

        ax.set_xticks(tick_idxs)
        ax.set_xticklabels(tick_labels, rotation=0)  # No rotation needed usually

        # Zero Line
        if 0 in full_range:
            zero_idx = full_range.index(0)
            ax.axvline(x=zero_idx, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label="Zero Error")

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Error (Frames)")
        if ax == axes[0]:
            ax.set_ylabel("Frequency")
        else:
            ax.set_ylabel("")
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_distribution_hist.png"))
    plt.close()

    # Error Distribution (ms) - LOOSE
    rep_fps = df_raw['FPS'].mode()[0] if not df_raw.empty else 60.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, event, title, color in zip(axes, events, titles, colors):
        subset = df_hist[df_hist["Type"] == event]

        if subset.empty:
            continue

        # Fill gaps (using frames as base to ensure uniform binning)
        vals = subset['ErrorFrames'].astype(int)
        min_val = vals.min()
        max_val = vals.max()

        plot_min = min(min_val, 0) - 2
        plot_max = max(max_val, 0) + 2
        full_range = list(range(plot_min, plot_max + 1))

        counts = vals.value_counts().reindex(full_range, fill_value=0)
        dist_data = counts.reset_index()
        dist_data.columns = ['ErrorFrames', 'Count']

        sns.barplot(
            data=dist_data, x="ErrorFrames", y="Count",
            ax=ax, color=color, edgecolor="black", alpha=0.8
        )

        # Ticks logic (from original code)
        bin_times = [f * (1000.0 / rep_fps) for f in full_range]
        min_time = min(bin_times)
        max_time = max(bin_times)

        targets = []
        curr = 0

        # Positive & Zero
        while curr <= max_time + 40:  # + buffer
            targets.append(curr)
            curr += 50

        # Negative
        curr = -50
        while curr >= min_time - 40:
            targets.append(curr)
            curr -= 50
        targets.sort()

        # Find the closest bin index for each target
        tick_idxs = []
        tick_labels = []

        for t in targets:
            closest_idx = min(range(len(bin_times)), key=lambda i: abs(bin_times[i] - t))

            # Avoid duplicate ticks if resolution is low
            if closest_idx not in tick_idxs:
                tick_idxs.append(closest_idx)
                tick_labels.append(str(t))

        ax.set_xticks(tick_idxs)
        ax.set_xticklabels(tick_labels)

        if 0 in full_range:
            zero_idx = full_range.index(0)
            ax.axvline(x=zero_idx, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label="Zero Error")

        ax.set_title(f"{title}", fontweight='bold')
        ax.set_xlabel("Error (ms)")
        if ax == axes[0]:
            ax.set_ylabel("Frequency")
        else:
            ax.set_ylabel("")
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_distribution_hist_ms.png"))
    plt.close()

    print(f"\n[Success] All results saved to: {out_dir}")
    print(f" - Reliability calculated with STRICT ({strict_tolerance_ms}ms) tolerance.")
    print(f" - Plots generated with LOOSE ({loose_tolerance_ms}ms) data.")


if __name__ == "__main__":
    run_benchmark()
