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

    def __str__(self):
        return self.value


@dataclass
class SupportPhase:
    support_type: SupportType
    start_frame: int
    end_frame: int
    duration: int
    valid: bool = True

    def __repr__(self):
        t_name = self.support_type.name if isinstance(self.support_type, SupportType) else str(self.support_type)
        return f"SupportPhase(type={t_name}, frames={self.start_frame}-{self.end_frame})"


class GaitEventType(str, Enum):
    TOE_OFF = "toe_off"
    HEEL_STRIKE = "heel_strike"

    def __str__(self):
        return self.value


@dataclass
class GaitEvent:
    frame: int
    event_type: GaitEventType


def compute_support_phases(left_phases: list[GaitPhase], right_phases: list[GaitPhase], total_frames: int) -> list[
    SupportPhase]:

    if total_frames == 0:
        return []

    # Masks for valid data
    known_left = np.zeros(total_frames, dtype=bool)
    known_right = np.zeros(total_frames, dtype=bool)

    # Stance phase mask
    stance_left = np.zeros(total_frames, dtype=bool)
    stance_right = np.zeros(total_frames, dtype=bool)

    def fill_masks(phases, known_mask, stance_mask):
        for ph in phases:
            # Index trimming
            s = max(0, ph.start_frame)
            e = min(total_frames - 1, ph.end_frame)
            if s > e: continue

            known_mask[s: e + 1] = True

            if ph.phase_type == PhaseType.STANCE:
                stance_mask[s: e + 1] = True

    fill_masks(left_phases, known_left, stance_left)
    fill_masks(right_phases, known_right, stance_right)

    # Valid data intersection (So we don't create imbalance)
    valid_intersection = known_left & known_right

    # Calculate support types
    support_labels = np.full(total_frames, SupportType.UNKNOWN, dtype=object)

    # Double Suppor
    double_mask = valid_intersection & stance_left & stance_right
    support_labels[double_mask] = SupportType.DOUBLE

    # Single Left
    single_left_mask = valid_intersection & stance_left & (~stance_right)
    support_labels[single_left_mask] = SupportType.SINGLE_LEFT

    # Single Right
    single_right_mask = valid_intersection & stance_right & (~stance_left)
    support_labels[single_right_mask] = SupportType.SINGLE_RIGHT

    # 5. RLE Compression into SupportPhase object
    phases = []
    start = 0
    current_type = support_labels[0]

    for i in range(1, total_frames):
        if support_labels[i] != current_type:
            if current_type != SupportType.UNKNOWN:
                phases.append(SupportPhase(current_type, start, i - 1, i - start))
            start = i
            current_type = support_labels[i]

    if current_type != SupportType.UNKNOWN:
        phases.append(SupportPhase(current_type, start, total_frames - 1, total_frames - start))

    return phases


def build_phases_from_events(events_dict: dict, valid_ranges: list[tuple], total_frames: int = None):

    # --- AUTO-CALCULATE TOTAL FRAMES ---
    if total_frames is None:
        if not valid_ranges:
            all_events = events_dict.get('left', []) + events_dict.get('right', [])
            max_ev = max(e.frame for e in all_events) if all_events else 0
            if max_ev == 0: return [], [], []
            total_frames = max_ev + 10
        else:
            max_end = max(end for _, end in valid_ranges)
            total_frames = max_end + 1

    # -----------------------------------

    def is_in_valid_range(start_frame, end_frame):
        # Whole interval must be in valid range
        for r_start, r_end in valid_ranges:
            if r_start <= start_frame and end_frame <= r_end:
                return True
        return False

    def build_phases_for_leg(events, leg: Leg) -> list[GaitPhase]:
        phases = []
        if not events: return phases

        sorted_events = sorted(events, key=lambda e: e.frame)

        for i in range(len(sorted_events) - 1):
            e1 = sorted_events[i]
            e2 = sorted_events[i + 1]

            p_type = PhaseType.UNKNOWN

            # HS -> TO = STANCE
            if e1.event_type == GaitEventType.HEEL_STRIKE and e2.event_type == GaitEventType.TOE_OFF:
                p_type = PhaseType.STANCE
            # TO -> HS = SWING
            elif e1.event_type == GaitEventType.TOE_OFF and e2.event_type == GaitEventType.HEEL_STRIKE:
                p_type = PhaseType.SWING
            else:
                # Two subsequent events are the same - double detection of the same event, ignore
                continue

            # Create phase if it's in valid range
            if is_in_valid_range(e1.frame, e2.frame):
                phases.append(GaitPhase(leg, p_type, e1.frame, e2.frame, e2.frame - e1.frame, valid=True))

        return phases

    l_events = events_dict.get('left', [])
    r_events = events_dict.get('right', [])

    l_phases = build_phases_for_leg(l_events, Leg.LEFT)
    r_phases = build_phases_for_leg(r_events, Leg.RIGHT)

    support_phases = compute_support_phases(l_phases, r_phases, total_frames)

    return l_phases, r_phases, support_phases
