"""Gap correction utilities for PPG analysis (optional functionality)."""

import numpy as np
from typing import Optional
from signal_processing import compute_threshold, nfillgap


def gap_correction(tk: np.ndarray) -> Optional[np.ndarray]:    
    """Correct gaps in RR intervals using adaptive thresholding."""
    # Threshold multipliers for upper and lower thresholds
    kupper = 1.5
    kupper_fine = 1 / kupper * 1.15
    klower = 1 / kupper * 0.8

    dtk = np.diff(tk)

    # Remove false positives
    threshold = compute_threshold(dtk)
    fp = dtk < 0.7 * threshold
    tk[np.where(fp)[0] + 1] = []
    tn = tk
    dtk = np.diff(tk)
    dtn = dtk

    # Gaps are detected by deviation from the median in difference series
    threshold = compute_threshold(dtk)
    gaps = np.where((dtk > threshold * kupper) & (dtk > 0.5))[0]
    if not gaps:
        return
    threshold_at_gap = threshold[gaps] * kupper

    # Gaps on first and last pulses are not allowed
    while gaps[0] < 1:
        tn[0] = []
        dtk[0] = []
        threshold[0] = []
        gaps = np.where(dtk > threshold * kupper)[0]
        threshold_at_gap = threshold[gaps] * kupper
        if not gaps:
            return
    while gaps[-1] > (dtk.size - 2):
        tn[-1] = []
        dtk[-1] = []
        threshold[-1] = []
        gaps = np.where(dtk > threshold * kupper)[0]
        threshold_at_gap = threshold[gaps] * kupper
        if not gaps:
            return

    nfill = 1  # Start filling with one sample
    while gaps:
        # In each iteration, try to fill with one more sample
        for kk in range(0, gaps.size):
            auxtn = nfillgap(tn, gaps, gaps[kk], nfill)
            auxdtn = np.diff(auxtn)

            correct = np.all(auxdtn[gaps[kk]:(gaps[kk]+nfill+1)] < kupper_fine * threshold_at_gap[kk])
            limit_exceeded = np.any([auxdtn[gaps[kk]:(gaps[kk]+nfill+1)] < klower * threshold_at_gap[kk],
                                     auxdtn[gaps[kk]:(gaps[kk]+nfill+1)] < 0.5])

            if limit_exceeded:
                # Check that lower theshold is not exceeded. Use previous nfill instead
                auxtn = nfillgap(tn, gaps, gaps[kk], nfill - 1)
                auxdtn = np.diff(auxtn)
                tn = auxtn
                gaps = gaps + nfill - 1
            elif correct:
                # If correct number of samples, save serie
                tn = auxtn
                gaps = gaps + nfill

        # Compute gaps for next iteration
        dtn = np.diff(tn)
        threshold = compute_threshold(dtn)
        gaps = np.where((dtn > (threshold * kupper)) & (dtn > 0.5))[0]
        threshold_at_gap = threshold[gaps] * kupper
        nfill = nfill + 1

    return tn
