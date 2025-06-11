"""HRV analysis utilities."""

import numpy as np
from typing import Dict
from signal_processing import compute_threshold


def time_metrics(tk: np.ndarray) -> Dict[str, int]:
    """Compute time domain HRV metrics."""
    rr = np.diff(tk)
    rr[rr == 0] = []  # Remove repeated beats

    threshold = compute_threshold(rr)
    dRR = np.diff(rr)  # (ms)
    rr[rr < 0.7*threshold] = np.nan
    rr[rr > 1.3*threshold] = np.nan

    # Compute time domain indices
    mhr = np.nanmean(60.0 / rr)  # (beats / min)
    sdnn = 1000 * np.nanstd(rr, ddof=0)  # (ms)
    rmssd = 1000 * np.sqrt(np.sum(np.square(dRR[~np.isnan(dRR)])) / dRR[~np.isnan(dRR)].size)  # (ms)
    sdsd = 1000 * np.nanstd(dRR, ddof=0)  # (ms)
    pnn50 = 100 * (np.sum(np.abs(dRR) > 0.05)) / np.sum(~np.isnan(dRR))  # (%)
    
    mhr = int(round(mhr))
    sdnn = int(round(sdnn))
    rmssd = int(round(rmssd))
    sdsd = int(round(sdsd))
    pnn50 = int(round(pnn50))

    results = {
        "MHR [beats/min]": mhr,
        "SDNN [ms]": sdnn,
        "RMSSD [ms]": rmssd,
        "SDSD [ms]": sdsd,
        "pNN50": pnn50
    }

    # Print metrics
    print("MHR: %d beats/min" % mhr)
    print("SDNN: %d ms" % sdnn)
    print("RMSSD: %d ms" % rmssd)
    print("SDSD: %d ms" % sdsd)
    print("pNN50: %d%%" % pnn50)

    return results
