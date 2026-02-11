"""
This module handles the generation of static visualization reports for gait analysis.

It reads the analysis results (JSON) and produces a suite of plots including:
- Pie charts for stance/swing ratios.
- Gantt charts for gait phases.
- Kinematic timelines and average cycles.
- Cyclograms for symmetry analysis.
- Center of Mass (CoM) vertical excursion timelines.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.logger import log

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.autolayout': False,
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
            "gantt_single_r": "#d32f2f",

            # Center of Mass line
            "com_line": "#8e24aa",
        }

    def generate_plots_from_json(self, analysis_source: str | dict):
        """
        Orchestrates the generation of all configured plots based on the provided JSON data.

        Args:
            analysis_source (str | dict): Path to the analysis results JSON file or the data dictionary itself.
        """
        data = {}
        source_name = "Data Dictionary"

        if isinstance(analysis_source, str):
            if not os.path.exists(analysis_source):
                return
            with open(analysis_source, 'r') as f:
                data = json.load(f)
            source_name = os.path.basename(analysis_source)
        elif isinstance(analysis_source, dict):
            data = analysis_source

        log("PLOTTER", f"Generating charts for {source_name}...", level="info")

        valid_ranges = data.get("metadata", {}).get("valid_ranges", [])

        # --- BASIC OVERVIEW ---
        if "spatiotemporal" in data:
            self._plot_phase_pie_charts(data["spatiotemporal"])

        if "gait_phases" in data:
            self._plot_gantt(data["gait_phases"], valid_ranges)

        if "kinematics_stats" in data:
            self._plot_avg_gait_cycles(data["kinematics_stats"])

        # --- DETAILED ANALYSIS ---
        if "raw_kinematics" in data and "gait_phases" in data:
            self._plot_leg_timeline_separated(data, "left")
            self._plot_leg_timeline_separated(data, "right")

        if "com_analysis" in data:
            self._plot_com_timeline(data["com_analysis"].get("raw_signal", []),
                                    data.get("gait_phases", {}).get("support", []), valid_ranges)

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
        fig, axes = plt.subplots(1, 3, figsize=(10, 4)) # Increased figsize slightly

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
                       wedgeprops=dict(width=0.4),
                       textprops={'fontsize': 11}) # Explicit text size for pies
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
                   wedgeprops=dict(width=0.4),
                   textprops={'fontsize': 11})
            ax.set_title("Support Ratio")
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')
            ax.axis('off')

        plt.tight_layout()
        # bbox_inches='tight' trims the white borders
        plt.savefig(os.path.join(self.output_dir, "01_phases_pie_triple.png"), dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def _plot_gantt(self, phases_data, valid_ranges):
        """
        Creates a compact Gantt chart showing the temporal progression of gait phases.
        """

        # If no valid ranges defined -> use full video
        if not valid_ranges:
            max_f = 0
            for side in ['left', 'right', 'support']:
                if phases_data.get(side):
                    max_f = max(max_f, max(p['end'] for p in phases_data[side]))
            valid_ranges = [(0, max_f)]

        fig, axes = self._create_broken_axis_fig(valid_ranges, figsize=(15, 3.5)) # Slightly wider/taller
        bar_h = 1.0

        for ax_idx, (ax, (v_start, v_end)) in enumerate(zip(axes, valid_ranges)):

            # Helper for drawing phases
            def draw_layer(phase_list, y_pos, color_fn):
                relevant = [p for p in phase_list if p['end'] >= v_start and p['start'] <= v_end]
                for p in relevant:
                    col = color_fn(p)
                    ax.broken_barh([(p['start'], p['duration'])], (y_pos - 0.5, bar_h),
                                   facecolors=col, edgecolor=None)

            draw_layer(phases_data.get('left', []), 2.5,
                       lambda p: self.colors["gantt_stance"] if "stance" in p['type'].lower() else self.colors[
                           "gantt_swing"])
            draw_layer(phases_data.get('right', []), 1.5,
                       lambda p: self.colors["gantt_stance"] if "stance" in p['type'].lower() else self.colors[
                           "gantt_swing"])

            def supp_col(p):
                t = p['type'].lower()
                if "double" in t: return self.colors["gantt_double"]
                if "left" in t: return self.colors["gantt_single_l"]
                if "right" in t: return self.colors["gantt_single_r"]
                return "#cccccc"

            draw_layer(phases_data.get('support', []), 0.5, supp_col)

            self._style_broken_axis_subplot(ax, ax_idx, len(valid_ranges), v_start, v_end)

        axes[0].set_yticks([0.5, 1.5, 2.5])
        axes[0].set_yticklabels(['Support', 'Right Leg', 'Left Leg'])

        fig.suptitle("Gait Phases Timeline", fontsize=16, y=1.05) # Moved title up slightly

        patches = [
            mpatches.Patch(color=self.colors['gantt_stance'], label='Stance'),
            mpatches.Patch(color=self.colors['gantt_swing'], label='Swing'),
            mpatches.Patch(color=self.colors['gantt_double'], label='Double Supp.'),
            mpatches.Patch(color=self.colors['gantt_single_l'], label='Single L'),
            mpatches.Patch(color=self.colors['gantt_single_r'], label='Single R')
        ]

        # Legend Position
        fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=5, frameon=False)

        # Adjust margins to give space for X-axis label and Legend
        plt.subplots_adjust(top=0.82, bottom=0.20, left=0.08, right=0.98)

        fig.text(0.5, 0.02, 'Frame Number', ha='center', fontsize=12)

        plt.savefig(os.path.join(self.output_dir, "01_gantt_chart.png"), dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def _plot_avg_gait_cycles(self, stats):
        """
        Plots the average kinematic trajectories.
        """
        joints = {"Hip": ["SHOULDER", "HIP", "KNEE"],
                  "Knee": ["HIP", "KNEE", "ANKLE"],
                  "Ankle": ["KNEE", "ANKLE", "FOOT"]}

        for j_name, kws in joints.items():
            fig, ax = plt.subplots(figsize=(6, 5)) # Slightly larger
            has_data = False

            for key, val in stats.items():
                if all(kw in key for kw in kws):
                    side = "left" if "LEFT" in key else "right"
                    m = np.array(val["mean"])
                    s = np.array(val["std"])
                    x = np.linspace(0, 100, len(m))

                    ax.plot(x, m, color=self.colors[side], label=side.capitalize(), lw=3) # Thicker line
                    ax.fill_between(x, m - s, m + s, color=self.colors[side], alpha=0.15)
                    has_data = True

            if has_data:
                ax.set_title(f"Avg {j_name} Cycle")
                ax.set_xlabel("% Gait Cycle")
                ax.set_ylabel("Angle (°)")
                ax.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"01_avg_cycle_{j_name.lower()}.png"), dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close()

    # =========================================================================
    # DETAILED ANALYSIS
    # =========================================================================

    def _plot_leg_timeline_separated(self, data, side):
        """
        Plots the raw kinematic angles timeline with phase background.
        """
        raw_kps = data.get("raw_kinematics", {})
        phases = data.get("gait_phases", {}).get(side, [])
        valid_ranges = data.get("metadata", {}).get("valid_ranges", [])

        # Fallback if no ranges defined -> use full video
        if not valid_ranges:
            total_len = len(list(raw_kps.values())[0])
            valid_ranges = [(0, total_len - 1)]

        keys_map = {
            "Hip": [k for k in raw_kps if "SHOULDER" in k and "HIP" in k and side.upper() in k],
            "Knee": [k for k in raw_kps if "HIP" in k and "KNEE" in k and "ANKLE" in k and side.upper() in k],
            "Ankle": [k for k in raw_kps if "KNEE" in k and "ANKLE" in k and "FOOT" in k and side.upper() in k]
        }

        for joint_name, joint_keys in keys_map.items():
            if not joint_keys: continue

            fig, axes = self._create_broken_axis_fig(valid_ranges, figsize=(14, 3.5))

            # Iterate over each "clip" (subplot)
            for ax_idx, (ax, (v_start, v_end)) in enumerate(zip(axes, valid_ranges)):

                # Background Phases
                # Filter only phases relevant to this clip
                clip_phases = [p for p in phases if p['end'] >= v_start and p['start'] <= v_end]
                for p in clip_phases:
                    start_rect = max(p['start'], v_start)
                    end_rect = min(p['end'], v_end)

                    col = self.colors["stance_bg"] if "stance" in p['type'].lower() else self.colors["swing_bg"]
                    ax.axvspan(start_rect, end_rect, color=col, alpha=0.7, lw=0)

                # Plot Signal Data
                for k in joint_keys:
                    # Slice signal
                    signal_full = np.array(raw_kps[k])

                    # Ensure bounds
                    s_idx = max(0, v_start)
                    e_idx = min(len(signal_full), v_end + 1)

                    if s_idx < e_idx:
                        signal_segment = signal_full[s_idx:e_idx]

                        x_frames = np.arange(s_idx, e_idx)
                        ax.plot(x_frames, signal_segment, color=self.colors[side], lw=2.0)

                self._style_broken_axis_subplot(ax, ax_idx, len(valid_ranges), v_start, v_end)

            fig.suptitle(f"{side.capitalize()} {joint_name} Angle", fontsize=16, y=0.98)
            axes[0].set_ylabel("Angle (°)")

            # Legend
            patches = [mpatches.Patch(color=self.colors["stance_bg"], label='Stance'),
                       mpatches.Patch(color=self.colors["swing_bg"], label='Swing')]
            axes[-1].legend(handles=patches, loc='upper right', frameon=True, framealpha=1.0)

            fig.text(0.5, 0.02, 'Frame Number', ha='center', fontsize=12)

            # Adjust bottom to make room for larger X label
            plt.subplots_adjust(top=0.85, bottom=0.22, right=0.98, left=0.06)
            plt.savefig(os.path.join(self.output_dir, f"02_detail_{side}_{joint_name.lower()}.png"), dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close()

    # =========================================================================
    # SYMMETRY
    # =========================================================================

    def _plot_symmetry_overlay(self, data, joint_name, keywords):
        """
        Plots the raw kinematic angles of both legs on the same graph.
        """
        raw_kps = data.get("raw_kinematics", {})
        phases = data.get("gait_phases", {}).get("support", [])
        valid_ranges = data.get("metadata", {}).get("valid_ranges", [])

        l_keys = [k for k in raw_kps if all(kw in k for kw in keywords) and "LEFT" in k]
        r_keys = [k for k in raw_kps if all(kw in k for kw in keywords) and "RIGHT" in k]
        if not l_keys or not r_keys: return

        if not valid_ranges:
            total_len = len(list(raw_kps.values())[0])
            valid_ranges = [(0, total_len - 1)]

        fig, axes = self._create_broken_axis_fig(valid_ranges, figsize=(14, 4.5)) # Taller for readability

        for ax_idx, (ax, (v_start, v_end)) in enumerate(zip(axes, valid_ranges)):

            # Background phases
            clip_phases = [p for p in phases if p['end'] >= v_start and p['start'] <= v_end]
            for p in clip_phases:
                s_r = max(p['start'], v_start)
                e_r = min(p['end'], v_end)

                stype = p['type'].lower()
                if "double" in stype: col = self.colors["double_bg"]
                elif "left" in stype: col = self.colors["single_left_bg"]
                elif "right" in stype: col = self.colors["single_right_bg"]
                else: col = "white"
                ax.axvspan(s_r, e_r, color=col, alpha=0.7, lw=0)

            # Signals
            for k in l_keys:
                full = np.array(raw_kps[k])
                s, e = max(0, v_start), min(len(full), v_end + 1)
                if s < e: ax.plot(np.arange(s, e), full[s:e], color=self.colors["left"],
                                  label="Left" if ax_idx == 0 and k == l_keys[0] else "", lw=2.5)

            for k in r_keys:
                full = np.array(raw_kps[k])
                s, e = max(0, v_start), min(len(full), v_end + 1)
                if s < e: ax.plot(np.arange(s, e), full[s:e], color=self.colors["right"],
                                  label="Right" if ax_idx == 0 and k == r_keys[0] else "", lw=2.5)

            self._style_broken_axis_subplot(ax, ax_idx, len(valid_ranges), v_start, v_end)

        fig.suptitle(f"{joint_name} Symmetry (Left vs Right)", fontsize=16, y=0.98)
        axes[0].set_ylabel("Angle (°)")

        # Legend
        handles = [mpatches.Patch(color=self.colors["left"], label="Left"),
                   mpatches.Patch(color=self.colors["right"], label="Right")]
        axes[-1].legend(handles=handles, loc="upper right", frameon=True, framealpha=1.0)

        fig.text(0.5, 0.02, 'Frame Number', ha='center', fontsize=12)

        plt.subplots_adjust(top=0.90, bottom=0.18, right=0.98, left=0.06)
        plt.savefig(os.path.join(self.output_dir, f"03_symmetry_{joint_name.lower()}.png"), dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # =========================================================================
    # CYCLOGRAMS
    # =========================================================================

    def _plot_cyclograms(self, stats):
        """
        Generates Angle-Angle plots (Cyclograms) for joint pairs (e.g., Hip-Knee, Knee-Ankle).
        Useful for analyzing inter-joint coordination and symmetry.
        """
        pairs = [("HIP", "KNEE", "Hip-Knee"), ("KNEE", "ANKLE", "Knee-Ankle")]

        # --- RANGES (HARD LIMITS) ---
        # If data is within this range, these limits are used
        # If data exceeds this range, the graph auto-expands (Soft Lock)
        std_ranges = {
            "Hip-Knee": {"x": [-20, 40], "y": [-10, 70]},
            "Knee-Ankle": {"x": [-10, 70], "y": [50, 100]}
        }

        def find_key(j_kw, s_kw):
            for k in stats:
                if j_kw == "HIP" and "SHOULDER" in k and "HIP" in k and "KNEE" in k and s_kw.upper() in k: return k
                if j_kw == "KNEE" and "HIP" in k and "KNEE" in k and "ANKLE" in k and s_kw.upper() in k: return k
                if j_kw == "ANKLE" and "KNEE" in k and "ANKLE" in k and "FOOT" in k and s_kw.upper() in k: return k
            return None

        for j1, j2, title in pairs:
            fig, ax = plt.subplots(figsize=(6, 6)) # Square figure
            has_data = False

            all_x, all_y = [], []

            for side in ['left', 'right']:
                k1, k2 = find_key(j1, side), find_key(j2, side)
                if k1 and k2:
                    v1 = np.array(stats[k1]['mean'])
                    v2 = np.array(stats[k2]['mean'])
                    # Close loop
                    v1 = np.append(v1, v1[0])
                    v2 = np.append(v2, v2[0])
                    ax.plot(v1, v2, label=side.capitalize(), color=self.colors[side], lw=3.0, alpha=0.8)
                    all_x.extend(v1); all_y.extend(v2)
                    has_data = True

            if has_data:
                # Axis scaling
                limits = std_ranges.get(title, {})
                target_x, target_y = limits.get("x", [-20, 80]), limits.get("y", [-20, 80])
                data_x_min, data_x_max = min(all_x), max(all_x)
                data_y_min, data_y_max = min(all_y), max(all_y)

                final_x_min = min(target_x[0], data_x_min - 5)
                final_x_max = max(target_x[1], data_x_max + 5)

                final_y_min = min(target_y[0], data_y_min - 5)
                final_y_max = max(target_y[1], data_y_max + 5)

                ax.set_xlim(final_x_min, final_x_max)
                ax.set_ylim(final_y_min, final_y_max)

                ax.set_xlabel(f"{j1} Angle (°)")
                ax.set_ylabel(f"{j2} Angle (°)")
                ax.set_title(f"{title} Cyclogram")
                ax.legend(frameon=True, framealpha=1.0)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"03_cyclogram_{j1.lower()}_{j2.lower()}.png"), dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close()

    # =========================================================================
    # CENTER OF MASS
    # =========================================================================

    def _plot_com_timeline(self, com_data, support_phases, valid_ranges):
        """
        Plots the vertical excursion of the Center of Mass (CoM) over time.
        Overlays background colors representing support phases (Single L/R, Double).
        """
        if not com_data: return

        if not valid_ranges:
            valid_ranges = [(0, len(com_data) - 1)]

        fig, axes = self._create_broken_axis_fig(valid_ranges, figsize=(12, 3))

        # Collector for valid data points to determine scaling
        all_valid_points = []

        for ax_idx, (ax, (v_start, v_end)) in enumerate(zip(axes, valid_ranges)):
            clip_phases = [p for p in support_phases if p['end'] >= v_start and p['start'] <= v_end]
            for p in clip_phases:
                stype = p['type'].lower()
                if "double" in stype:
                    col = self.colors["double_bg"]
                elif "left" in stype:
                    col = self.colors["single_left_bg"]
                elif "right" in stype:
                    col = self.colors["single_right_bg"]
                else:
                    col = "white"

                # Clip rect to view
                s_draw = max(p['start'], v_start)
                e_draw = min(p['end'], v_end)
                ax.axvspan(s_draw, e_draw, color=col, alpha=0.5, lw=0)

            # Signal Plotting
            full_sig = np.array(com_data)
            s_idx = max(0, v_start)
            e_idx = min(len(full_sig), v_end + 1)

            if s_idx < e_idx:
                segment = full_sig[s_idx:e_idx]

                # Collect valid points for auto-scale logic
                valid_seg = [v for v in segment if v is not None]
                all_valid_points.extend(valid_seg)
                ax.plot(np.arange(s_idx, e_idx), segment, color=self.colors["com_line"], lw=2.5)

            self._style_broken_axis_subplot(ax, ax_idx, len(valid_ranges), v_start, v_end)

        # Smart scaling
        # Default window
        target_min, target_max = 90, 130

        if all_valid_points:
            data_min, data_max = min(all_valid_points), max(all_valid_points)
            final_min = min(target_min, data_min - 2)
            final_max = max(target_max, data_max + 2)

            axes[0].set_ylim(final_min, final_max)
        else:
            axes[0].set_ylim(target_min, target_max)

        fig.suptitle("Center of Mass Vertical Excursion", fontsize=16, y=0.98)
        axes[0].set_ylabel("% Leg Length")

        # Legend
        handles = [mpatches.Patch(color=self.colors["double_bg"], label="Double Supp."),
                   mpatches.Patch(color=self.colors["single_left_bg"], label="Single L"),
                   mpatches.Patch(color=self.colors["single_right_bg"], label="Single R")]
        axes[-1].legend(handles=handles, loc="upper right", frameon=True, framealpha=1.0)

        fig.text(0.5, 0.02, 'Frame Number', ha='center', fontsize=12)
        plt.subplots_adjust(top=0.88, bottom=0.22, right=0.98, left=0.06)
        plt.savefig(os.path.join(self.output_dir, "02_detail_com_height.png"), dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _create_broken_axis_fig(valid_ranges, figsize=(12, 3)):
        """Initializes figure and axes for broken-axis plots."""
        durations = [end - start for start, end in valid_ranges]
        if sum(durations) == 0: durations = [1] * len(durations)

        fig, axes = plt.subplots(1, len(valid_ranges), sharey=True, figsize=figsize,
                                 gridspec_kw={'width_ratios': durations, 'wspace': 0.15})

        if len(valid_ranges) == 1: axes = [axes]
        return fig, axes

    @staticmethod
    def _style_broken_axis_subplot(ax, ax_idx, total_plots, v_start, v_end):
        """Applies visual styling (spines, diagonal slashes) for broken axes."""
        ax.set_xlim(v_start, v_end)

        # Hide spines between plots
        if total_plots > 1:
            if ax_idx < total_plots - 1:
                ax.spines['right'].set_visible(False)
                ax.tick_params(right=False)

            if ax_idx > 0:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_ticks_position('none')

        # Add diagonal slash to indicate break
        d = .015
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)

        if ax_idx < total_plots - 1:
            ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        if ax_idx > 0:
            ax.plot((-d, +d), (-d, +d), **kwargs)
            ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
