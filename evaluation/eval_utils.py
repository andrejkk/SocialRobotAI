"""
Shared evaluation utilities.
Imported by evaluation.py (CLI) and k-fold-cross-evaluation.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def expand_instantaneous_events(events_df, tolerance=0.5):
    """
    Expand instantaneous events (time_start == time_end) to tolerance windows
    [time - tolerance, time + tolerance].  Interval events are unchanged.
    """
    expanded = events_df.copy()
    mask = expanded['time_start'] == expanded['time_end']
    expanded.loc[mask, 'time_start'] -= tolerance
    expanded.loc[mask, 'time_end']   += tolerance
    return expanded


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_events(gt_df, pred_df, eval_start_time=None, instantaneous_tolerance=0.5):
    """
    Evaluate event detection using temporal overlap (TP/FP/FN in seconds).

    Returns a dict with:
      tp, fp, fn,
      macro_precision, macro_recall, macro_f1,
      micro_precision, micro_recall, micro_f1,
      eID_metrics, comparisons,
      precision, recall, f1  (aliases for macro, for backward compatibility)
    """
    gt_df   = gt_df.copy()
    pred_df = pred_df.copy()

    # Normalize eID types to string, stripping ".0" from float-encoded integers
    # (e.g. pandas reads integer Excel columns as float64: 405.0 → "405")
    def _norm_eid(val):
        try:
            return str(int(float(val)))
        except (ValueError, TypeError):
            return str(val)

    gt_df['eID']   = gt_df['eID'].map(_norm_eid)
    pred_df['eID'] = pred_df['eID'].map(_norm_eid)

    gt_has_instant   = (gt_df['time_start']   == gt_df['time_end']).any()
    pred_has_instant = (pred_df['time_start'] == pred_df['time_end']).any()

    if gt_has_instant or pred_has_instant:
        n_gt   = (gt_df['time_start']   == gt_df['time_end']).sum()
        n_pred = (pred_df['time_start'] == pred_df['time_end']).sum()
        print(f"\nInstantaneous events detected (±{instantaneous_tolerance}s tolerance).")
        if gt_has_instant:   print(f"  GT:        {n_gt} instantaneous events expanded")
        if pred_has_instant: print(f"  Predicted: {n_pred} instantaneous events expanded")

    gt_df   = expand_instantaneous_events(gt_df,   tolerance=instantaneous_tolerance)
    pred_df = expand_instantaneous_events(pred_df, tolerance=instantaneous_tolerance)

    if eval_start_time is not None:
        gt_df   = gt_df[gt_df['time_end']     >= eval_start_time].reset_index(drop=True)
        pred_df = pred_df[pred_df['time_end'] >= eval_start_time].reset_index(drop=True)
        print(f"\nFiltering by eval_start_time={eval_start_time}s: "
              f"{len(gt_df)} GT, {len(pred_df)} predicted events remain")

    total_tp = total_fp = total_fn = 0.0
    comparisons = []
    eID_metrics = {}

    all_eids = set(gt_df['eID'].unique()) | set(pred_df['eID'].unique())

    for eid in all_eids:
        gt_evs   = gt_df[gt_df['eID']     == eid].reset_index(drop=True)
        pred_evs = pred_df[pred_df['eID'] == eid].reset_index(drop=True)

        eid_tp = eid_fp = eid_fn = 0.0
        used_pred = set()

        for _, gt_ev in gt_evs.iterrows():
            gs, ge = gt_ev['time_start'], gt_ev['time_end']
            best_idx, best_overlap = None, 0

            for pidx, pred_ev in pred_evs.iterrows():
                ps, pe = pred_ev['time_start'], pred_ev['time_end']
                overlap = max(0, min(ge, pe) - max(gs, ps))
                if overlap > best_overlap:
                    best_overlap, best_idx = overlap, pidx

            if best_idx is not None and best_idx not in used_pred:
                pred_ev  = pred_evs.iloc[best_idx]
                ps, pe   = pred_ev['time_start'], pred_ev['time_end']
                tp = max(0.0, min(ge, pe) - max(gs, ps))
                fp = max(0.0, (pe - ps) - tp)
                fn = max(0.0, (ge - gs) - tp)

                pair_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                pair_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
                pair_f1   = (2 * pair_prec * pair_rec / (pair_prec + pair_rec)
                             if (pair_prec + pair_rec) > 0 else 0)

                comparisons.append({
                    'eID': eid, 'gt_start': gs, 'gt_end': ge,
                    'pred_start': ps, 'pred_end': pe,
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'precision': pair_prec, 'recall': pair_rec, 'f1': pair_f1,
                })
                total_tp += tp; eid_tp += tp
                total_fp += fp; eid_fp += fp
                total_fn += fn; eid_fn += fn
                used_pred.add(best_idx)
            else:
                fn_miss = ge - gs
                total_fn += fn_miss; eid_fn += fn_miss

        for pidx, pred_ev in pred_evs.iterrows():
            if pidx not in used_pred:
                fp_extra = pred_ev['time_end'] - pred_ev['time_start']
                total_fp += fp_extra; eid_fp += fp_extra

        eID_metrics[eid] = {'tp': eid_tp, 'fp': eid_fp, 'fn': eid_fn}

    # MACRO metrics (duration-weighted)
    macro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    macro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    macro_f1   = (2 * macro_prec * macro_rec / (macro_prec + macro_rec)
                  if (macro_prec + macro_rec) > 0 else 0)

    # MICRO metrics (per-eID average)
    # Classes with no ground-truth support in this set (tp + fn == 0) are
    # prediction-only false positives.  Their recall is undefined, so including
    # them as a 0 would unfairly drag the per-class average down.  We exclude
    # them from the micro average (matching sklearn's behaviour) while still
    # counting their predicted time as FP in the macro metrics above.
    micro_precs, micro_recs = [], []
    for m in eID_metrics.values():
        has_gt_support = (m['tp'] + m['fn']) > 0
        if not has_gt_support:
            continue
        p = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0
        r = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) > 0 else 0
        micro_precs.append(p); micro_recs.append(r)

    micro_prec = np.mean(micro_precs) if micro_precs else 0
    micro_rec  = np.mean(micro_recs)  if micro_recs  else 0
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                  if (micro_prec + micro_rec) > 0 else 0)

    return {
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'macro_precision': macro_prec, 'macro_recall': macro_rec, 'macro_f1': macro_f1,
        'micro_precision': micro_prec, 'micro_recall': micro_rec, 'micro_f1': micro_f1,
        'eID_metrics': eID_metrics, 'comparisons': comparisons,
        # backward-compat aliases
        'precision': macro_prec, 'recall': macro_rec, 'f1': macro_f1,
    }


# ---------------------------------------------------------------------------
# Timing-difference metrics
# ---------------------------------------------------------------------------

def compute_timing_differences(comparisons):
    """
    From the list of matched comparisons, compute start-time and end-time
    differences (predicted − ground truth) and label-match flags.

    Returns a list of dicts with keys:
      eID, gt_start, gt_end, pred_start, pred_end,
      start_diff, end_diff, label_match
    """
    diffs = []
    for c in comparisons:
        diffs.append({
            'eID':        c['eID'],
            'gt_start':   c['gt_start'],
            'gt_end':     c['gt_end'],
            'pred_start': c['pred_start'],
            'pred_end':   c['pred_end'],
            'start_diff': c['pred_start'] - c['gt_start'],
            'end_diff':   c['pred_end']   - c['gt_end'],
            'label_match': True,  # matching is done per-eID
        })
    return diffs


def print_timing_differences(diffs):
    """Print per-pair and aggregate timing-difference metrics."""
    if not diffs:
        print("\nNo matched event pairs — timing-difference metrics unavailable.")
        return

    start_diffs = np.array([d['start_diff'] for d in diffs])
    end_diffs   = np.array([d['end_diff']   for d in diffs])

    # print("\n" + "=" * 60)
    # print("Timing-Difference Metrics  (predicted − ground truth)")
    # print("=" * 60)

    # for d in diffs:
    #     print(f"  eID={d['eID']}  "
    #           f"GT[{d['gt_start']:.2f}–{d['gt_end']:.2f}]  "
    #           f"Pred[{d['pred_start']:.2f}–{d['pred_end']:.2f}]  "
    #           f"Δstart={d['start_diff']:+.4f}s  "
    #           f"Δend={d['end_diff']:+.4f}s  "
    #           f"label_match={d['label_match']}")

    print(f"\n  Matched pairs: {len(diffs)}")
    print(f"  Start-time differences  — mean: {np.mean(start_diffs):+.4f}s  "
          f"std: {np.std(start_diffs):.4f}s  "
          f"min(abs): {np.min(np.abs(start_diffs)):.4f}s  max(abs): {np.max(np.abs(start_diffs)):.4f}s")
    print(f"  End-time   differences  — mean: {np.mean(end_diffs):+.4f}s  "
          f"std: {np.std(end_diffs):.4f}s  "
          f"min(abs): {np.min(np.abs(end_diffs)):.4f}s  max(abs): {np.max(np.abs(end_diffs)):.4f}s")


def plot_timing_histograms(diffs, output_path='timing_diff_histogram.png'):
    """Plot histograms of start-time and end-time differences and save to PNG."""
    if not diffs:
        print("No matched pairs — skipping timing-difference histogram.")
        return

    start_diffs = np.array([d['start_diff'] for d in diffs])
    end_diffs   = np.array([d['end_diff']   for d in diffs])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Start-time differences
    s_mean, s_std = np.mean(start_diffs), np.std(start_diffs)
    axes[0].hist(start_diffs, bins='auto', edgecolor='black', alpha=0.7)
    axes[0].axvline(s_mean, color='red', linestyle='--',
                    label=f'mean={s_mean:+.3f}s')
    axes[0].axvspan(s_mean - s_std, s_mean + s_std, alpha=0.2, color='orange',
                    label=f'±std={s_std:.3f}s')
    axes[0].set_title('Start-Time Differences (pred − GT)')
    axes[0].set_xlabel('Difference (s)')
    axes[0].set_ylabel('Count')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # End-time differences
    e_mean, e_std = np.mean(end_diffs), np.std(end_diffs)
    axes[1].hist(end_diffs, bins='auto', edgecolor='black', alpha=0.7)
    axes[1].axvline(e_mean, color='red', linestyle='--',
                    label=f'mean={e_mean:+.3f}s')
    axes[1].axvspan(e_mean - e_std, e_mean + e_std, alpha=0.2, color='orange',
                    label=f'±std={e_std:.3f}s')
    axes[1].set_title('End-Time Differences (pred − GT)')
    axes[1].set_xlabel('Difference (s)')
    axes[1].set_ylabel('Count')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Timing-difference histogram saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_signals_with_events(
    sigs_df, gt_events_df, pred_events_df,
    t_int=None, sigs_lst=None, event_defs=None,
    output_path='events_evaluation_plot.png', window_size_s=60
):
    """
    Plot signals with GT (solid fills) and predicted (hatched fills) events.
    Creates one pair of subplots (GT / Predicted) per signal per 60-s window.
    """
    if sigs_lst is None:
        sigs_lst = [c for c in sigs_df.columns if c.startswith('sig_')][:1]
    if t_int is None:
        t_int = [sigs_df['time_s'].min(), sigs_df['time_s'].max()]

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    all_eids = list(set(gt_events_df['eID'].unique()) | set(pred_events_df['eID'].unique()))
    event_colors = {eid: colors[i % len(colors)] for i, eid in enumerate(all_eids)}

    time_windows = []
    cur = t_int[0]
    while cur < t_int[1]:
        time_windows.append((cur, min(cur + window_size_s, t_int[1])))
        cur += window_size_s

    n_sigs    = len(sigs_lst)
    n_windows = len(time_windows)
    n_subplots = n_sigs * n_windows * 2
    fig, axes = plt.subplots(n_subplots, 1,
                             figsize=(14, 2.5 * max(n_subplots, 1)))
    if n_subplots == 1:
        axes = [axes]

    for si, sig in enumerate(sigs_lst):
        for wi, (ws, we) in enumerate(time_windows):
            gt_ax   = axes[si * n_windows * 2 + wi * 2]
            pred_ax = axes[si * n_windows * 2 + wi * 2 + 1]

            mask = (sigs_df['time_s'] >= ws) & (sigs_df['time_s'] <= we)
            win  = sigs_df[mask]

            # --- GT panel ---
            gt_ax.plot(win['time_s'], win[sig], 'b-', linewidth=1.5, label='Signal')
            gt_seen = set()
            for _, ev in gt_events_df.iterrows():
                should = True
                if event_defs and ev['eID'] in event_defs:
                    should = sig in event_defs[ev['eID']]['sigs']
                if should and ev['time_end'] >= ws and ev['time_start'] <= we:
                    color = event_colors[ev['eID']]
                    lbl = str(ev['eID']) if ev['eID'] not in gt_seen else ''
                    gt_seen.add(ev['eID'])
                    gt_ax.axvspan(ev['time_start'], ev['time_end'],
                                  alpha=0.3, color=color, label=lbl)
            gt_ax.set_xlim(ws, we)
            gt_ax.set_ylabel(sig)
            gt_ax.set_title(f'GT: {sig} [{ws:.1f}-{we:.1f}s]')
            gt_ax.legend(loc='upper right', fontsize=7)
            gt_ax.grid(True, alpha=0.3)

            # --- Predicted panel ---
            pred_ax.plot(win['time_s'], win[sig], 'b-', linewidth=1.5, label='Signal')
            pred_seen = set()
            for _, ev in pred_events_df.iterrows():
                should = True
                if event_defs and ev['eID'] in event_defs:
                    should = sig in event_defs[ev['eID']]['sigs']
                if should and ev['time_end'] >= ws and ev['time_start'] <= we:
                    color = event_colors.get(ev['eID'], 'gray')
                    lbl = str(ev['eID']) if ev['eID'] not in pred_seen else ''
                    pred_seen.add(ev['eID'])
                    pred_ax.axvspan(ev['time_start'], ev['time_end'],
                                    alpha=0.3, color=color, hatch='///', label=lbl)
            pred_ax.set_xlim(ws, we)
            pred_ax.set_xlabel('Time (s)')
            pred_ax.set_ylabel(sig)
            pred_ax.set_title(f'Predicted: {sig} [{ws:.1f}-{we:.1f}s]')
            pred_ax.legend(loc='upper right', fontsize=7)
            pred_ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()
