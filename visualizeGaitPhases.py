import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from gaitStructs import PhaseType, SupportType, Leg


def print_statistics(left_phases, right_phases, support_phases):
    def phase_summary(phases, leg):
        stance = [p.duration for p in phases if p.phase_type == PhaseType.STANCE and p.valid]
        swing  = [p.duration for p in phases if p.phase_type == PhaseType.SWING and p.valid]
        total  = sum(stance) + sum(swing)
        stance_pct = (sum(stance) / total * 100) if total else 0
        swing_pct  = (sum(swing) / total * 100) if total else 0
        return {
            "avg_stance": np.mean(stance) if stance else 0,
            "avg_swing": np.mean(swing) if swing else 0,
            "stance_pct": stance_pct,
            "swing_pct": swing_pct
        }

    def support_summary(supports):
        single_l = [s.duration for s in supports if s.support_type == SupportType.SINGLE_LEFT and s.valid]
        single_r = [s.duration for s in supports if s.support_type == SupportType.SINGLE_RIGHT and s.valid]
        double   = [s.duration for s in supports if s.support_type == SupportType.DOUBLE and s.valid]
        total = sum(single_l) + sum(single_r) + sum(double)
        return {
            "avg_single_l": np.mean(single_l) if single_l else 0,
            "avg_single_r": np.mean(single_r) if single_r else 0,
            "avg_double": np.mean(double) if double else 0,
            "pct_single_l": (sum(single_l)/total*100) if total else 0,
            "pct_single_r": (sum(single_r)/total*100) if total else 0,
            "pct_double": (sum(double)/total*100) if total else 0
        }

    # --- Detailed phases ---
    print("\n=== INDIVIDUAL PHASES ===")
    for phases, label in [(left_phases, "Left"), (right_phases, "Right")]:
        for p in phases:
            if not p.valid:
                continue
            print(f"{label:<5} | {p.phase_type.value:<6} | {p.start_frame:>5} -> {p.end_frame:<5} | dur={p.duration:<4}")

    # --- Leg summaries ---
    left_stats = phase_summary(left_phases, Leg.LEFT)
    right_stats = phase_summary(right_phases, Leg.RIGHT)

    print("\n=== LEG SUMMARY ===")
    print(f"Left  - Avg stance: {left_stats['avg_stance']:.1f}  Avg swing: {left_stats['avg_swing']:.1f}  "
          f"Stance%: {left_stats['stance_pct']:.1f}%  Swing%: {left_stats['swing_pct']:.1f}%")
    print(f"Right - Avg stance: {right_stats['avg_stance']:.1f}  Avg swing: {right_stats['avg_swing']:.1f}  "
          f"Stance%: {right_stats['stance_pct']:.1f}%  Swing%: {right_stats['swing_pct']:.1f}%")

    # --- Support phases ---
    print("\n=== SUPPORT PHASES ===")
    for s in support_phases:
        if not s.valid:
            continue
        print(f"{s.support_type.value:<13} | {s.start_frame:>5} -> {s.end_frame:<5} | dur={s.duration:<4}")

    # --- Support summary ---
    sup_stats = support_summary(support_phases)
    print("\n=== SUPPORT SUMMARY ===")
    print(f"Avg single L: {sup_stats['avg_single_l']:.1f}  ({sup_stats['pct_single_l']:.1f}%)")
    print(f"Avg single R: {sup_stats['avg_single_r']:.1f}  ({sup_stats['pct_single_r']:.1f}%)")
    print(f"Avg double:   {sup_stats['avg_double']:.1f}    ({sup_stats['pct_double']:.1f}%)")

    print("\n" + "=" * 50)


def visualize_gait_phases(left_phases, right_phases, support_phases, x_start=0, x_end=None):
    phase_colors = {
        PhaseType.STANCE: 'red',
        PhaseType.SWING: 'green'
    }
    phase_labels = {
        PhaseType.STANCE: 'stance',
        PhaseType.SWING: 'swing'
    }

    support_colors = {
        SupportType.SINGLE_LEFT: 'orange',
        SupportType.SINGLE_RIGHT: 'orange',
        SupportType.DOUBLE: 'blue'
    }
    support_labels = {
        SupportType.SINGLE_LEFT: 'single (L)',
        SupportType.SINGLE_RIGHT: 'single (R)',
        SupportType.DOUBLE: 'double support'
    }

    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    # LEFT LEG
    for phase in left_phases:
        if not phase.valid or phase.phase_type == PhaseType.UNKNOWN:
            continue  # <- přeskakuje UNKNOWN
        axs[0].axvspan(
            phase.start_frame,
            phase.end_frame,
            color=phase_colors[phase.phase_type],
            alpha=0.5
        )
    axs[0].set_ylabel("Left Foot")
    axs[0].set_yticks([])
    axs[0].set_title("Left Foot Gait Phases")
    axs[0].legend(
        handles=[Patch(color=c, label=l) for c, l in zip(phase_colors.values(), phase_labels.values())],
        loc='upper right'
    )

    # RIGHT LEG
    for phase in right_phases:
        if not phase.valid or phase.phase_type == PhaseType.UNKNOWN:
            continue
        axs[1].axvspan(
            phase.start_frame,
            phase.end_frame,
            color=phase_colors[phase.phase_type],
            alpha=0.5
        )
    axs[1].set_ylabel("Right Foot")
    axs[1].set_yticks([])
    axs[1].set_title("Right Foot Gait Phases")
    axs[1].legend(
        handles=[Patch(color=c, label=l) for c, l in zip(phase_colors.values(), phase_labels.values())],
        loc='upper right'
    )

    # SUPPORT PHASES
    for sp in support_phases:
        if getattr(sp, "support_type", None) not in support_colors:
            continue  # <- přeskočí UNKNOWN a cokoli jiného
        axs[2].axvspan(
            sp.start_frame,
            sp.end_frame,
            color=support_colors[sp.support_type],
            alpha=0.5
        )

    axs[2].set_yticks([])
    axs[2].set_xlabel('Frame index')
    axs[2].set_title('Support Phases')
    axs[2].legend(
        handles=[Patch(color=c, label=l) for c, l in zip(support_colors.values(), support_labels.values())],
        loc='upper right'
    )
    axs[2].set_xlim(x_start, x_end)

    plt.tight_layout()
    plt.show()

