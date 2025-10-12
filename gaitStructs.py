from enum import Enum
from dataclasses import dataclass

import numpy as np


class Leg(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class PhaseType(str, Enum):
    STANCE = "stance"
    SWING = "swing"
    UNKNOWN = "unknown"


@dataclass
class GaitPhase:
    leg: Leg
    phase_type: PhaseType
    start_frame: int
    end_frame: int
    duration: int
    valid: bool


class SupportType(str, Enum):
    SINGLE_LEFT = "single_left"
    SINGLE_RIGHT = "single_right"
    DOUBLE = "double_support"
    UNKNOWN = "unknown"


@dataclass
class SupportPhase:
    support_type: SupportType
    start_frame: int
    end_frame: int
    duration: int
    valid: bool = True


class GaitEventType(str, Enum):
    TOE_OFF = "toe_off"
    HEEL_STRIKE = "heel_strike"


@dataclass
class GaitEvent:
    frame: int
    event_type: GaitEventType


def filter_phases_generic(phases, excluded_ranges):
    for ph in phases:
        if any(not (ph.end_frame < s or ph.start_frame > e) for s, e in excluded_ranges):
            ph.valid = False
    return [ph for ph in phases if ph.valid]


def compute_support_phases(left_phases, right_phases, total_frames):
    stance_left = np.zeros(total_frames, dtype=bool)
    stance_right = np.zeros(total_frames, dtype=bool)
    valid_left = np.zeros(total_frames, dtype=bool)
    valid_right = np.zeros(total_frames, dtype=bool)

    # Left leg
    for ph in left_phases:
        if not ph.valid:
            continue
        valid_left[ph.start_frame:ph.end_frame + 1] = True
        if ph.phase_type == PhaseType.STANCE:
            stance_left[ph.start_frame:ph.end_frame + 1] = True

    # Right leg
    for ph in right_phases:
        if not ph.valid:
            continue
        valid_right[ph.start_frame:ph.end_frame + 1] = True
        if ph.phase_type == PhaseType.STANCE:
            stance_right[ph.start_frame:ph.end_frame + 1] = True

    # Mask out valid indices
    valid_mask = valid_left & valid_right

    total_stance = stance_left.astype(int) + stance_right.astype(int)
    support_labels = np.array([SupportType.UNKNOWN for _ in range(total_frames)], dtype=object)
    support_labels[valid_mask & (total_stance == 2)] = SupportType.DOUBLE
    support_labels[valid_mask & (total_stance == 1) & stance_left] = SupportType.SINGLE_LEFT
    support_labels[valid_mask & (total_stance == 1) & stance_right] = SupportType.SINGLE_RIGHT

    # Create phases
    phases = []
    start = 0
    current_type = support_labels[0]
    for i in range(1, total_frames):
        if support_labels[i] != current_type:
            phases.append(SupportPhase(current_type, start, i - 1, i - start))
            start = i
            current_type = support_labels[i]

    phases.append(SupportPhase(current_type, start, total_frames - 1, total_frames - start))

    # Filter out unknown data
    return [ph for ph in phases if ph.support_type != SupportType.UNKNOWN]


def build_phases_from_events(left_events, right_events, excluded_ranges, total_frames, start_offset):

    def in_excluded(i: int) -> bool:
        return any(start <= i <= end for start, end in excluded_ranges)

    def build_phases_for_leg(events: list[GaitEvent], leg: Leg) -> list[GaitPhase]:
        phases = []
        current_phase = PhaseType.UNKNOWN
        phase_start = None

        for event in sorted(events, key=lambda e: e.frame):
            i = event.frame

            # Skip excluded regions
            if in_excluded(i):
                if current_phase != PhaseType.UNKNOWN:
                    # End current phase as invalid
                    phases.append(GaitPhase(
                        leg, current_phase,
                        phase_start + start_offset,
                        i - 1 + start_offset,
                        i - phase_start,
                        valid=False
                    ))
                    current_phase = PhaseType.UNKNOWN
                    phase_start = None
                continue

            # Event to phase translation
            if event.event_type == GaitEventType.HEEL_STRIKE:
                new_phase = PhaseType.STANCE
            elif event.event_type == GaitEventType.TOE_OFF:
                new_phase = PhaseType.SWING
            else:
                new_phase = PhaseType.UNKNOWN

            # Start of first phase or change
            if new_phase != current_phase:
                # End previous phase
                if current_phase != PhaseType.UNKNOWN and phase_start is not None:
                    valid = not any(not (i - 1 < s or phase_start > e) for s, e in excluded_ranges)
                    phases.append(GaitPhase(
                        leg, current_phase,
                        phase_start + start_offset,
                        i - 1 + start_offset,
                        i - phase_start,
                        valid
                    ))
                # Start new phase
                current_phase = new_phase
                phase_start = i

        # Close last phase
        if current_phase != PhaseType.UNKNOWN and phase_start is not None:
            valid = not any(not (total_frames - 1 < s or phase_start > e) for s, e in excluded_ranges)
            phases.append(GaitPhase(
                leg, current_phase,
                phase_start + start_offset,
                total_frames - 1 + start_offset,
                total_frames - phase_start,
                valid
            ))

        return [p for p in phases if p.valid]

    l_phases = build_phases_for_leg(left_events, Leg.LEFT)
    r_phases = build_phases_for_leg(right_events, Leg.RIGHT)
    support_phases = compute_support_phases(l_phases, r_phases, total_frames)

    return l_phases, r_phases, support_phases
