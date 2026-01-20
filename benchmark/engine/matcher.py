def match_events_greedy(gt_frames, pred_frames, tolerance_frames, verbose=False, label=""):
    """
    Matches predicted events to ground truth events using a greedy strategy.

    For each ground truth event, it finds the closest predicted event within the
    tolerance window. If multiple predictions are candidates, the one with the
    smallest absolute error is chosen. Each prediction can only be used once.

    Args:
        gt_frames (list[int]): List of frame indices for ground truth events.
        pred_frames (list[int]): List of frame indices for predicted events.
        tolerance_frames (int): Maximum allowed frame difference for a match.
        verbose (bool): If True, prints detailed matching logs.
        label (str): Label prefix for verbose logging (e.g., "Left Heel Strike").

    Returns:
        tuple: A tuple containing:
            - errors (list[int]): List of frame differences (pred - gt) for valid matches.
            - misses (int): Number of unmatched ground truth events (False Negatives).
            - extras (int): Number of unmatched predictions (False Positives).
    """
    gt_frames = sorted(gt_frames)
    pred_frames = sorted(pred_frames)

    used_predictions = set()
    errors = []
    matched_gt_count = 0

    if verbose:
        print(f"\n--- DEBUG MATCHING: {label} (Tol: {tolerance_frames}) ---")
        print(f"GT List:   {gt_frames}")
        print(f"Pred List: {pred_frames}")

    for gt in gt_frames:
        candidates = []
        for i, pred in enumerate(pred_frames):
            if i in used_predictions: continue

            dist = pred - gt
            # If in range, add to candidates
            if abs(dist) <= tolerance_frames:
                candidates.append((i, dist, pred))

            # Early exit if out of range
            if pred > gt + tolerance_frames:
                break

        if candidates:
            # Pick the closest candidate
            best_i, best_err, best_pred = min(candidates, key=lambda x: abs(x[1]))

            errors.append(best_err)
            used_predictions.add(best_i)
            matched_gt_count += 1

            if verbose:
                print(f"  ✅ MATCH: GT {gt:<5} <--> Pred {best_pred:<5} | Diff: {best_err:+d}")
        else:
            if verbose:
                print(f"  ❌ MISS:  GT {gt:<5} <--> (No candidate in range)")

    misses = len(gt_frames) - matched_gt_count
    extras = len(pred_frames) - len(used_predictions)

    if verbose:
        # Print extra detections that were not matched
        for i, pred in enumerate(pred_frames):
            if i not in used_predictions:
                print(f"  ⚠️ EXTRA: (No GT)   <--> Pred {pred:<5}")
        print("-" * 40)

    return errors, misses, extras
