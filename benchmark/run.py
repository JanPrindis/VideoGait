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

    # Evaluation Loop Containers
    raw_matches = []
    reliability_stats = {
        "left": {et.value: {"TP": 0, "FN": 0, "FP": 0} for et in [GaitEventType.HEEL_STRIKE, GaitEventType.TOE_OFF]},
        "right": {et.value: {"TP": 0, "FN": 0, "FP": 0} for et in [GaitEventType.HEEL_STRIKE, GaitEventType.TOE_OFF]}
    }

    tolerance_ms = cfg['evaluation']['tolerance_ms']
    print(f"\n--- Starting Benchmark: {experiment_name} ---")
    print(f"[Eval] Tolerance: {tolerance_ms} ms")

    debug_active = args.debug

    for i, item in enumerate(test_files):
        print(f"[{i + 1}/{len(test_files)}] {item['name']}...", end="\r")

        # Predict
        pred_result = wrapper.predict(item['kp'])
        if not pred_result: continue

        current_fps = pred_result.get('framerate', global_fps)
        tolerance_frames = int((tolerance_ms / 1000.0) * current_fps)
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

                should_print = debug_active and (i == 0)
                errors, misses, extras = match_events_greedy(
                    gt_valid,
                    predictions,
                    tolerance_frames,
                    verbose=should_print,
                    label=f"{item['name']} | {side} | {et.name}"
                )

                reliability_stats[side][et.value]["TP"] += len(errors)
                reliability_stats[side][et.value]["FN"] += misses
                reliability_stats[side][et.value]["FP"] += extras

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

    # --- PROCESS & SAVE DATA ---
    df_raw = pd.DataFrame(raw_matches)
    if df_raw.empty:
        print("No matches found! Check tolerance or data.")
        return
    df_raw.to_csv(os.path.join(out_dir, "raw_matches.csv"), index=False)

    # Stats Summary
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
    stats_summary.to_csv(os.path.join(out_dir, "stats_summary.csv"))

    # Reliability Report
    rel_rows = []
    for side in ['left', 'right']:
        for et_val, counts in reliability_stats[side].items():
            tp, fn, fp = counts["TP"], counts["FN"], counts["FP"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            rel_rows.append({
                "Side": side, "Type": et_val,
                "TP": tp, "FN": fn, "FP": fp,
                "Total_GT": tp + fn,
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1_Score": round(f1, 4)
            })

    df_rel = pd.DataFrame(rel_rows)
    df_rel.to_csv(os.path.join(out_dir, "reliability_report.csv"), index=False)

    # --- PLOTS ---
    # Reliability (TP/FN/FP)
    df_rel_agg = df_rel.groupby("Type")[["TP", "FN", "FP"]].sum().reset_index()
    df_rel_melted = df_rel_agg.melt(id_vars="Type", value_vars=["TP", "FN", "FP"],
                                    var_name="Category", value_name="Count")

    plt.figure(figsize=(5, 5))
    sns.barplot(
        data=df_rel_melted, x="Type", y="Count", hue="Category",
        palette={"TP": "#2ecc71", "FN": "#e74c3c", "FP": "#f39c12"},
        edgecolor="black", linewidth=0.5, width=0.7
    )
    plt.title(f"Reliability", fontweight='bold')
    plt.ylabel("Count")
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title=None, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reliability_barplot.png"))

    # Accuracy Boxplot (Frames)
    plt.figure(figsize=(4, 5))
    ax = sns.boxplot(
        data=df_raw, x="Type", y="ErrorFrames",
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5, "markerfacecolor": "gray"},  # Decentní tečky
        width=0.5,
        boxprops=dict(facecolor='#3498db', alpha=0.8)
    )
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.title("Accuracy (Frames)", fontweight='bold')
    plt.ylabel("Error (Frames)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_boxplot_frames.png"))

    # Accuracy Boxplot (ms)
    plt.figure(figsize=(4, 5))
    sns.boxplot(
        data=df_raw, x="Type", y="ErrorMs",
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5, "markerfacecolor": "gray"},
        width=0.5,
        boxprops=dict(facecolor='#9b59b6', alpha=0.8)
    )
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.title("Accuracy (Time)", fontweight='bold')
    plt.ylabel("Error (ms)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_boxplot_ms.png"))

    # Error Distribution
    df_hist = df_raw.copy()

    # Get X range
    min_err = int(df_hist['ErrorFrames'].min())
    max_err = int(df_hist['ErrorFrames'].max())
    x_range = list(range(min_err, max_err + 1))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    events = [GaitEventType.HEEL_STRIKE.value, GaitEventType.TOE_OFF.value]
    titles = ["Heel Strike", "Toe Off"]
    colors = ["#3498db", "#e74c3c"]

    max_count = 0

    for ax, event, title, color in zip(axes, events, titles, colors):
        subset = df_hist[df_hist["Type"] == event]

        # Count occurrences
        dist_data = subset.groupby('ErrorFrames').size().reset_index(name='Count')

        if dist_data.empty:
            continue

        sns.barplot(
            data=dist_data, x="ErrorFrames", y="Count",
            ax=ax,
            color=color, edgecolor="black", alpha=0.8
        )

        # Find bar for value 0
        if 0 in dist_data['ErrorFrames'].values:
            zero_idx = dist_data.index[dist_data['ErrorFrames'] == 0][0]

            # Draw vertical line representing 0 error
            ax.axvline(x=zero_idx, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label="Zero Error")

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Error (Frames)")
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Force integer ticks on X axis if range is small
        if len(x_range) < 20:
            ax.set_xticks(range(len(dist_data)))
            ax.set_xticklabels(dist_data['ErrorFrames'].astype(int))

        if ax == axes[0]:
            ax.set_ylabel("Frequency")
        else:
            ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_distribution_hist.png"))

    print(f"\n[Success] All results saved to: {out_dir}")
    print(f" - Plots generated with FULL DATA (Outliers visible)")


if __name__ == "__main__":
    run_benchmark()
