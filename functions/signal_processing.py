"""Signal processing utilities for PPG analysis."""

import numpy as np
import math
from scipy import signal
from scipy.interpolate import PchipInterpolator, CubicSpline
from typing import Tuple


def filtering_and_normalization(sig: np.ndarray, sig_fs: float) -> np.ndarray:
    """Apply filtering and normalization to the signal."""
    b, a = signal.butter(3, 0.3, btype='highpass', fs=sig_fs)  # type: ignore
    sig_filtered = signal.filtfilt(b, a, sig)
    b, a = signal.butter(3, 10, btype='lowpass', fs=sig_fs)  # type: ignore
    sig_filtered = signal.filtfilt(b, a, sig_filtered)
    sig_filtered = normalize(sig_filtered)
    return sig_filtered


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize signal to zero mean and unit variance."""
    return (x - np.mean(x)) / np.std(x)


def remove_impulse_artifacts(sig: np.ndarray) -> np.ndarray:
    """Remove impulse artifacts from the signal."""
    # Square of second derivative
    aux = np.diff(np.diff(sig)) ** 2
    aux = np.insert(aux, 0, aux[0])
    aux = np.append(aux, aux[-1])

    # Median filter threshold
    wind = 999
    if aux.size < wind:
        wind = aux.size
        if (wind % 2) != 1:
            wind = wind - 1
    mf = signal.medfilt(aux, wind)

    # Find impulses
    margin = 20
    impulses = np.asarray(np.where(aux > mf + 0.005)).ravel()
    for impulse in impulses:
        impulses = np.append(impulses, np.arange(impulse - margin, impulse + margin + 1))
    impulses = np.sort(impulses)
    impulses = np.unique(impulses)
    impulses = impulses[impulses >= 0]

    # Remove impulses
    output = sig
    output[impulses] = np.nan
    return output


def ppg_pulse_detection(sig: np.ndarray, sig_fs: float, fine_search: bool) -> np.ndarray:
    """Detect pulses in PPG signal."""
    # Linear-phase FIR filter
    ntaps = 3 * sig_fs + 1  # order + 1
    lpd_fp = 7.9
    lpd_fc = 8
    bands = [0, lpd_fp, lpd_fc, 0.5 * sig_fs]
    desired = [1, 0]
    b = signal.remez(ntaps, bands=bands, desired=desired, fs=sig_fs, type='differentiator')
    sig_filtered = signal.lfilter(b, [1], sig)

    delay = math.ceil((b.size - 1) / 2)
    sig_filtered = sig_filtered[delay:]
    sig_filtered = np.append(sig_filtered, np.zeros((delay, 1)))

    # Threshold delineator
    alfa = 0.2
    refract = 250e-03
    tao_rr = 1
    thr_incidences = 1.5
    n_d_int, threshold = adaptive_thresholding(sig_filtered, sig_fs, alfa, refract, tao_rr, thr_incidences)

    if fine_search:
        n_d = np.empty(len(n_d_int))
        n_d.fill(np.nan)
        fsi = 1000  # Hz
        t = np.arange(0, (len(sig) - 1) / sig_fs + 1 / sig_fs, 1 / sig_fs)
        w_nA = 250e-3
        w_nB = 150e-3
        n_d_int[n_d_int < 0] = 0
        wdw_n_d1 = n_d_int - round((w_nB / 2) * sig_fs)
        wdw_n_d2 = n_d_int + round((w_nA / 2) * sig_fs)
        for peak in range(len(n_d_int)):
            try:
                aux_t_i_nD = np.arange(t[wdw_n_d1[peak]], t[wdw_n_d2[peak]], 1 / fsi)
                cs = CubicSpline(t[wdw_n_d1[peak]:wdw_n_d2[peak]], sig_filtered[wdw_n_d1[peak]:wdw_n_d2[peak]])
                aux_ppg_i_nD = cs(aux_t_i_nD)
                pos_n_D = np.argmax(aux_ppg_i_nD)
                n_d[peak] = aux_t_i_nD[pos_n_D]
            except (ValueError, IndexError):
                pass
    else:
        n_d = np.divide(n_d_int, sig_fs)

    return n_d


def adaptive_thresholding(sig_filt: np.ndarray, sig_fs: float, alfa: float, refract: float, tao_rr: float, thr_incidences: float) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptive thresholding algorithm for pulse detection."""
    refract = int(round(refract*sig_fs))
    n_d = np.empty(0)
    peaks_added = np.empty(0)
    cond_vec = np.empty(0)
    sig_filtered_notnan = np.asarray(np.where(~np.isnan(sig_filt))).ravel()
    thres_ini_w_ini = sig_filtered_notnan[0]
    thres_ini_w_end = thres_ini_w_ini + np.round(10 * sig_fs)
    aux = sig_filt[thres_ini_w_ini:thres_ini_w_end+1]
    thres_ini = 3 * np.nanmean(aux[aux >= 0])
    thres = np.empty(sig_filt.shape)
    thres[:] = np.nan
    n = np.arange(1, sig_filt.size)
    rr = int(np.round(.75 * sig_fs))

    if (1 + rr) < sig_filt.size:
        thres[0:(1+rr)] = thres_ini - (thres_ini * (1 - alfa) / rr) * (n[0:(rr+1)] - 1)
        thres[rr:] = alfa * thres_ini
    else:
        thres[:] = thres_ini - (thres_ini * (1 - alfa) / rr) * (n - 1)

    kk = 1
    while True:
        cross_u_aux = np.asarray(np.where(sig_filt[(kk-1):] > thres[(kk-1):])).ravel()
        if cross_u_aux.size < 1:
            # No more pulses -> end
            break
        cross_u = kk - 1 + cross_u_aux[0]  # Next point to cross the actual threshold (down->up)

        cross_d_aux = np.asarray(np.where(sig_filt[(cross_u-1):] < thres[(cross_u-1):])).ravel()
        cross_d_aux = np.delete(cross_d_aux, np.where(cross_d_aux == 0))
        if cross_d_aux.size < 1:
            # No more pulses -> end
            break
        cross_d = cross_u - 1 + cross_d_aux[0]  # Next point to cross the actual threshold (up->down)

        if not cross_d:
            # No more pulses -> end
            break

        # Pulse detected
        vmax = np.max(sig_filt[cross_u:cross_d+1])
        imax = np.argmax(sig_filt[cross_u:cross_d+1])
        p = cross_u + imax
        n_d = np.append(n_d, p)
        peaks_filt_orig = n_d
        npeaks = n_d.size

        if npeaks > 3:
            tk_c = peaks_filt_orig[-1]
            tk1_c = peaks_filt_orig[-2]
            tk2_c = peaks_filt_orig[-3]
            cond = np.absolute((2 * tk1_c - tk2_c - tk_c) / ((tk1_c - tk2_c) * (tk_c - tk1_c) * (tk_c - tk2_c)))
            cond_vec = np.append(cond_vec, cond)
            if cond >= thr_incidences / (sig_fs * sig_fs):
                tk = int(n_d[-1])
                tk1 = int(n_d[-2])
                tk2 = int(n_d[-3])
                tk3 = int(n_d[-4])

                # Inserting a beat between tk2 and tk1
                aux_15 = sig_filt[tk2:(tk1+1)]
                aux_locs = signal.find_peaks(aux_15)[0]
                aux_peaks = aux_15[aux_locs]
                if aux_locs.size > 0:
                    aux_locs = aux_locs[aux_peaks >= 0.5 * np.max(aux_peaks)]
                    if aux_locs.size > 0:
                        aux_loc = np.argmin(np.absolute(aux_locs - aux_15.size) / 2)
                        tk15 = tk2 - 1 + aux_locs[aux_loc]
                    else:
                        tk15 = np.nan
                else:
                    tk15 = np.nan

                # Inserting a beat between tk1 and tk
                aux_05 = sig_filt[tk1:(tk+1)]
                aux_locs = signal.find_peaks(aux_05)[0]
                aux_peaks = aux_05[aux_locs]
                if aux_locs.size > 0:
                    aux_locs = aux_locs[aux_peaks >= 0.5 * np.max(aux_peaks)]
                    if aux_locs.size > 0:
                        aux_loc = np.argmin(np.absolute(aux_locs - aux_05.size) / 2)
                        tk05 = tk1 - 1 + aux_locs[aux_loc]
                    else:
                        tk05 = np.nan
                else:
                    tk05 = np.nan

                # Condition removing previous detection (cond1)
                cond1 = np.absolute((2 * tk2 - tk3 - tk1) / ((tk2 - tk3) * (tk1 - tk2) * (tk1 - tk3)))

                # Condition removing previous detection (cond2)
                cond2 = np.absolute((2 * tk2 - tk3 - tk) / ((tk2 - tk3) * (tk - tk2) * (tk - tk3)))

                # Condition adding a new detection between tk2 and tk1 (cond3)
                if not np.isnan(tk15):
                    cond3 = np.absolute((2 * tk1 - tk15 - tk) / ((tk1 - tk15) * (tk - tk1) * (tk - tk15)))
                else:
                    cond3 = np.inf

                # Condition adding a new detection between tk1 and tk (cond4)
                if not np.isnan(tk05):
                    np.seterr(divide='ignore')
                    cond4 = np.absolute((2 * tk05 - tk1 - tk) / ((tk05 - tk1) * (tk - tk05) * (tk - tk1)))
                else:
                    cond4 = np.inf

                high_cond = np.argmin(np.array([cond1, cond2, cond3, cond4]))
                match high_cond:
                    case 1:  # Best is to remove current detection
                        n_d = n_d[0:-1]
                        cond_vec = cond_vec[0:-1]
                        kk = cross_d
                        continue
                    case 2:  # Best is to remove previous detection
                        imax = np.argmax(sig_filt[(cross_u-refract):(cross_d+1)])
                        vmax = sig_filt[imax]
                        if not imax == 1:
                            p = cross_u - refract + imax
                        n_d = np.append(n_d[0:-2], p)
                        cond_vec = cond_vec[0:-1]
                        npeaks = npeaks - 1
                    case 3:  # Best is to add a detection between tk2 and tk1
                        peaks_added = np.append(peaks_added, tk15)
                    case 4:  # Best is to add a detection between tk1 and tk
                        peaks_added = np.append(peaks_added, tk05)

        # Update threshold
        n_rr_estimation = 3
        n_ampli_est = 3
        if npeaks >= n_rr_estimation + 1:
            rr = np.round(np.median(np.diff(n_d[-(n_rr_estimation+1):])))
        elif npeaks >= 2:
            rr = np.round(np.mean(np.diff(n_d)))
        kk = int(np.min(np.append(p + refract, sig_filt.size)))
        thres[p:(kk+1)] = vmax

        vfall = vmax * alfa
        if npeaks >= (n_ampli_est + 1):
            ampli_est = np.median(sig_filt[n_d[-(n_ampli_est+1):-1].astype(int)])
            if vmax >= (2 * ampli_est):
                vfall = alfa * ampli_est
                vmax = ampli_est

        fall_end = int(np.round(tao_rr * rr))
        if (kk + fall_end) < sig_filt.size:
            thres[kk:(kk+fall_end+1)] = vmax - (vmax - vfall) / fall_end * (n[kk:(kk+fall_end+1)] - kk)
            thres[(kk+fall_end):] = vfall
        else:
            thres[kk:] = vmax - (vmax - vfall) / fall_end * (n[kk:] - kk)

    n_d = np.unique(np.concatenate([n_d, peaks_added])).astype(int)
    return n_d, thres


def compute_threshold(rr: np.ndarray) -> np.ndarray:
    """Compute adaptive threshold for RR intervals."""
    wind = 29
    if rr.size < wind:
        wind = rr.size
        if (wind % 2) != 1:
            wind = wind - 1
    mf = signal.medfilt(np.concatenate((np.flipud(rr[0:wind // 2]), rr, np.flipud(rr[-(wind // 2):])))[:], wind)
    mf[mf > 1.5] = 1.5
    return mf[(wind // 2):-(wind // 2)]


def nfillgap(tk: np.ndarray, gaps: np.ndarray, current_gap: int, nfill: int) -> np.ndarray:
    """Fill gaps in RR intervals using interpolation."""
    dtk = np.diff(tk)
    gaps = np.delete(gaps, np.where(gaps == current_gap))
    dtk[gaps] = np.nan
    gap = dtk[current_gap]
    previousIntervals = dtk[np.amax([0, (current_gap - 20)]):current_gap]
    posteriorIntervals = dtk[(current_gap+1):np.amin([dtk.size, (current_gap + 21)])]
    npre = previousIntervals.size
    npos = posteriorIntervals.size

    pi = PchipInterpolator(np.concatenate([np.arange(0, npre), np.arange((nfill+npre+1), nfill+npre+npos+1)]),
                           np.concatenate([previousIntervals, posteriorIntervals]))
    intervals = pi(np.arange(npre, npre+nfill+1))
    intervals = intervals[0:-1] * gap / np.nansum(intervals)  # map intervals to gap
    return np.concatenate([tk[0:current_gap+1], tk[current_gap] + np.cumsum(intervals), tk[current_gap+1:]])
