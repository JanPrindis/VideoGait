"""
This module handles the generation of static visualization reports for gait analysis.

It reads the analysis results (JSON) and produces a suite of plots including:
- Pie charts for stance/swing ratios.
- Gantt charts for gait phases.
- Kinematic timelines and average cycles.
- Cyclograms for symmetry analysis.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.autolayout': True,
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.facecolor': 'white'
})


class GaitPlotter:
    """
    Generates all visual reports from analysis.json.
    Categorized into Basic, Detailed, and Symmetry plots.
    """

    def __init__(self, output_dir):
        """
        Initializes the plotter.

        Args:
            output_dir (str): Directory where generated plots will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # --- COLOR PALETTE ---
        self.colors = {
            "left": "#004ba0",
            "right": "#c62828",

            # Pie Charts
            "pie_stance": "#81c784",
            "pie_swing": "#ffb74d",
            "pie_single": "#64b5f6",
            "pie_double": "#ba68c8",

            # Timeline Backgrounds (Overlay)
            "stance_bg": "#a5d6a7",
            "swing_bg": "#ffe0b2",
            "double_bg": "#e1bee7",
            "single_left_bg": "#bbdefb",
            "single_right_bg": "#ffcdd2",

            # Gantt Chart Bars
            "gantt_stance": "#43a047",
            "gantt_swing": "#fb8c00",
            "gantt_double": "#8e24aa",
            "gantt_single_l": "#1976d2",
            "gantt_single_r": "#d32f2f"
        }

    def generate_plots_from_json(self, json_path: str):
        """
        Orchestrates the generation of all configured plots based on the provided JSON data.

        Args:
            json_path (str): Path to the analysis results JSON file.
        """
        if not os.path.exists(json_path): return

        with open(json_path, 'r') as f:
            data = json.load(f)

        print(f"[Plotter] Generating charts for {os.path.basename(json_path)}...")

        # --- BASIC OVERVIEW ---
        if "spatiotemporal" in data:
            self._plot_phase_pie_charts(data["spatiotemporal"])

        if "gait_phases" in data:
            self._plot_gantt_compact(data["gait_phases"])

        if "kinematics_stats" in data:
            self._plot_avg_gait_cycles(data["kinematics_stats"])

        # --- DETAILED ANALYSIS ---
        if "raw_kinematics" in data and "gait_phases" in data:
            self._plot_leg_timeline_separated(data, "left")
            self._plot_leg_timeline_separated(data, "right")

        # --- SYMMETRY ---
        if "kinematics_stats" in data:
            self._plot_cyclograms(data["kinematics_stats"])

        if "raw_kinematics" in data:
            self._plot_symmetry_overlay(data, "Hip", ["SHOULDER", "HIP", "KNEE"])
            self._plot_symmetry_overlay(data, "Knee", ["HIP", "KNEE", "ANKLE"])
            self._plot_symmetry_overlay(data, "Ankle", ["KNEE", "ANKLE", "FOOT"])

    # =========================================================================
    # BASIC PLOTS
    # =========================================================================

    def _plot_phase_pie_charts(self, spatio):
        """
        Generates pie charts visualizing the Stance/Swing ratio for each leg
        and the Single/Double support ratio.
        """
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        # Left and Right
        for ax, side in zip(axes[:2], ['left', 'right']):
            stats = spatio.get(side, {})
            st = stats.get('stance_time_avg', 0)
            sw = stats.get('swing_time_avg', 0)

            if st + sw > 0:
                ax.pie([st, sw],
                       labels=[f'Stance\n{st:.2f}s', f'Swing\n{sw:.2f}s'],
                       colors=[self.colors['pie_stance'], self.colors['pie_swing']],
                       autopct='%1.0f%%', startangle=90, pctdistance=0.85,
                       wedgeprops=dict(width=0.4))
                ax.set_title(f"{side.capitalize()} Leg")
            else:
                ax.text(0.5, 0.5, "No Data", ha='center')
                ax.axis('off')

        # Support Ratio
        ax = axes[2]
        supp = spatio.get('support_ratio', {})
        sin = supp.get('single_total_s', 0)
        dbl = supp.get('double_total_s', 0)

        if sin + dbl > 0:
            ax.pie([sin, dbl],
                   labels=[f'Single\n{sin:.1f}s', f'Double\n{dbl:.1f}s'],
                   colors=[self.colors['pie_single'], self.colors['pie_double']],
                   autopct='%1.0f%%', startangle=90, pctdistance=0.85,
                   wedgeprops=dict(width=0.4))
            ax.set_title("Support Ratio")
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "01_phases_pie_triple.png"), dpi=150)
        plt.close()

    def _plot_gantt_compact(self, phases_data):
        """
        Creates a compact Gantt chart showing the temporal progression of gait phases
        (Stance/Swing) for both legs and the resulting support phases.
        """
        fig, ax = plt.subplots(figsize=(14, 2.5))
        bar_h = 1.0

        # Helper to plot bars
        def plot_bars(phase_list, y_center, color_map_fn):
            for p in phase_list:
                col = color_map_fn(p)
                ax.broken_barh([(p['start'], p['duration'])],
                               (y_center - 0.5, bar_h),
                               facecolors=col, edgecolor=None)

        # Left Leg (Top)
        plot_bars(phases_data.get('left', []), 2.5,
                  lambda p: self.colors["gantt_stance"] if "stance" in p['type'].lower() else self.colors[
                      "gantt_swing"])

        # Right Leg (Middle)
        plot_bars(phases_data.get('right', []), 1.5,
                  lambda p: self.colors["gantt_stance"] if "stance" in p['type'].lower() else self.colors[
                      "gantt_swing"])

        # Support (Bottom)
        def get_support_color(p):
            stype = p['type'].lower()
            if "double" in stype: return self.colors["gantt_double"]
            if "left" in stype: return self.colors["gantt_single_l"]
            if "right" in stype: return self.colors["gantt_single_r"]
            return "#cccccc"

        plot_bars(phases_data.get('support', []), 0.5, get_support_color)

        # Formatting
        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(['Support', 'Right Leg', 'Left Leg'])
        ax.set_xlabel('Frame Number')

        # Set limits
        all_ends = [p['end'] for p in phases_data.get('support', [])]
        if all_ends: ax.set_xlim(0, max(all_ends))

        # Legend
        patches = [
            mpatches.Patch(color=self.colors['gantt_stance'], label='Stance'),
            mpatches.Patch(color=self.colors['gantt_swing'], label='Swing'),
            mpatches.Patch(color=self.colors['gantt_double'], label='Double Supp.'),
            mpatches.Patch(color=self.colors['gantt_single_l'], label='Single Left'),
            mpatches.Patch(color=self.colors['gantt_single_r'], label='Single Right')
        ]

        ax.legend(handles=patches, loc='lower center',
                  bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False)

        plt.savefig(os.path.join(self.output_dir, "01_gantt_chart.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_avg_gait_cycles(self, stats):
        """
        Plots the average kinematic trajectories (Hip, Knee, Ankle angles) over a normalized gait cycle (0-100%).
        Includes standard deviation shading.
        """
        joints = {"Hip": ["SHOULDER", "HIP", "KNEE"],
                  "Knee": ["HIP", "KNEE", "ANKLE"],
                  "Ankle": ["KNEE", "ANKLE", "FOOT"]}

        for j_name, kws in joints.items():
            fig, ax = plt.subplots(figsize=(5, 4))
            has_data = False

            for key, val in stats.items():
                if all(kw in key for kw in kws):
                    side = "left" if "LEFT" in key else "right"
                    m = np.array(val["mean"])
                    s = np.array(val["std"])
                    x = np.linspace(0, 100, len(m))

                    ax.plot(x, m, color=self.colors[side], label=side.capitalize(), lw=2)
                    ax.fill_between(x, m - s, m + s, color=self.colors[side], alpha=0.15)
                    has_data = True

            if has_data:
                ax.set_title(f"Avg {j_name} Cycle")
                ax.set_xlabel("% Gait Cycle")
                ax.set_ylabel("Angle (°)")
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"01_avg_cycle_{j_name.lower()}.png"), dpi=150)
            plt.close()

    # =========================================================================
    # DETAILED ANALYSIS
    # =========================================================================

    def _plot_leg_timeline_separated(self, data, side):
        """
        Plots the raw kinematic angles for a specific leg over time, overlaid with
        background colors representing the detected gait phases.
        """
        raw_kps = data.get("raw_kinematics", {})
        phases = data.get("gait_phases", {}).get(side, [])
        valid_ranges = data.get("metadata", {}).get("valid_ranges", [])

        keys_map = {
            "Hip": [k for k in raw_kps if "SHOULDER" in k and "HIP" in k and side.upper() in k],
            "Knee": [k for k in raw_kps if "HIP" in k and "KNEE" in k and "ANKLE" in k and side.upper() in k],
            "Ankle": [k for k in raw_kps if "KNEE" in k and "ANKLE" in k and "FOOT" in k and side.upper() in k]
        }

        total_len = len(list(raw_kps.values())[0])
        frames = np.arange(total_len)

        # Create validity mask
        mask = np.full(total_len, np.nan)
        if not valid_ranges:
            mask[:] = 1
        else:
            for s, e in valid_ranges:
                mask[max(0, s):min(total_len, e + 1)] = 1

        for joint_name, joint_keys in keys_map.items():
            if not joint_keys: continue

            fig, ax = plt.subplots(figsize=(12, 3))

            # Draw Phases Background
            for p in phases:
                col = self.colors["stance_bg"] if "stance" in p['type'].lower() else self.colors["swing_bg"]
                ax.axvspan(p['start'], p['end'], color=col, alpha=0.7, lw=0)

            # Draw Signal
            for k in joint_keys:
                ax.plot(frames, np.array(raw_kps[k]) * mask,
                        color=self.colors[side], label=f"{side.capitalize()} {joint_name}")

            ax.set_ylabel("Angle (°)")
            ax.set_xlabel("Frame Number")
            ax.set_title(f"{side.capitalize()} {joint_name} Angle")

            patches = [mpatches.Patch(color=self.colors["stance_bg"], label='Stance'),
                       mpatches.Patch(color=self.colors["swing_bg"], label='Swing')]
            ax.legend(handles=patches, loc='upper right', frameon=True, framealpha=1.0)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"02_detail_{side}_{joint_name.lower()}.png"), dpi=150)
            plt.close()

    # =========================================================================
    # SYMMETRY
    # =========================================================================

    def _plot_symmetry_overlay(self, data, joint_name, keywords):
        """
        Plots the raw kinematic angles of both legs on the same graph to visualize symmetry.
        Includes background coloring for support phases (Single L/R, Double).
        """
        raw_kps = data.get("raw_kinematics", {})
        phases = data.get("gait_phases", {}).get("support", [])
        valid_ranges = data.get("metadata", {}).get("valid_ranges", [])

        l_keys = [k for k in raw_kps if all(kw in k for kw in keywords) and "LEFT" in k]
        r_keys = [k for k in raw_kps if all(kw in k for kw in keywords) and "RIGHT" in k]

        if not l_keys or not r_keys: return

        fig, ax = plt.subplots(figsize=(12, 4))
        frames = np.arange(len(list(raw_kps.values())[0]))
        mask = np.full(len(frames), np.nan)

        if not valid_ranges:
            mask[:] = 1
        else:
            for s, e in valid_ranges:
                mask[max(0, s):min(len(frames), e + 1)] = 1

        # Draw Support Background
        for p in phases:
            stype = p['type'].lower()
            if "double" in stype:
                col = self.colors["double_bg"]
            elif "left" in stype:
                col = self.colors["single_left_bg"]
            elif "right" in stype:
                col = self.colors["single_right_bg"]
            else:
                col = "white"
            ax.axvspan(p['start'], p['end'], color=col, alpha=0.7, lw=0)

        # Draw Lines
        for k in l_keys:
            ax.plot(frames, np.array(raw_kps[k]) * mask, color=self.colors["left"], label="Left", lw=2)
        for k in r_keys:
            ax.plot(frames, np.array(raw_kps[k]) * mask, color=self.colors["right"], label="Right", lw=2)

        ax.set_title(f"{joint_name} Symmetry")
        ax.set_ylabel("Angle (°)")
        ax.legend(loc="upper right", frameon=True, framealpha=1.0)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"03_symmetry_{joint_name.lower()}.png"), dpi=150)
        plt.close()

    def _plot_cyclograms(self, stats):
        """
        Generates Angle-Angle plots (Cyclograms) for joint pairs (e.g., Hip-Knee, Knee-Ankle).
        Useful for analyzing inter-joint coordination and symmetry.
        """
        pairs = [("HIP", "KNEE", "Hip-Knee"), ("KNEE", "ANKLE", "Knee-Ankle")]

        def find_key(j_kw, s_kw):
            for k in stats:
                if j_kw == "HIP" and "SHOULDER" in k and "HIP" in k and "KNEE" in k and s_kw.upper() in k: return k
                if j_kw == "KNEE" and "HIP" in k and "KNEE" in k and "ANKLE" in k and s_kw.upper() in k: return k
                if j_kw == "ANKLE" and "KNEE" in k and "ANKLE" in k and "FOOT" in k and s_kw.upper() in k: return k
            return None

        for j1, j2, title in pairs:
            fig, ax = plt.subplots(figsize=(5, 5))
            has_data = False

            for side in ['left', 'right']:
                k1, k2 = find_key(j1, side), find_key(j2, side)
                if k1 and k2:
                    v1 = np.array(stats[k1]['mean'])
                    v2 = np.array(stats[k2]['mean'])
                    # Close the loop
                    v1 = np.append(v1, v1[0])
                    v2 = np.append(v2, v2[0])

                    ax.plot(v1, v2, label=side.capitalize(), color=self.colors[side], lw=2.5, alpha=0.8)
                    has_data = True

            if has_data:
                ax.set_xlabel(f"{j1} Angle (°)")
                ax.set_ylabel(f"{j2} Angle (°)")
                ax.set_title(f"{title} Cyclogram")
                ax.legend(frameon=True, framealpha=1.0)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"03_cyclogram_{j1.lower()}_{j2.lower()}.png"), dpi=150)
            plt.close()
