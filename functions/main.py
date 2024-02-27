import os
import firebase_admin
import numpy as np
import math
import json
from firebase_functions import storage_fn
from firebase_admin import credentials, storage
from firebase_functions.options import MemoryOption
from matplotlib import pyplot as plt
from scipy import signal
from scipy.interpolate import PchipInterpolator, CubicSpline
from configparser import InterpolationError
from google.cloud import storage as gcs
from uuid import uuid4

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'bsicos-app.appspot.com'})
storage_client = gcs.Client()
bucket = storage_client.bucket('bsicos-app.appspot.com')

# noinspection PyTupleAssignmentBalance
def filtering_and_normalization(sig, sig_fs):
    b, a = signal.butter(3, 0.3, btype='highpass', fs=sig_fs)
    sig_filtered = signal.filtfilt(b, a, sig)
    b, a = signal.butter(3, 10, btype='lowpass', fs=sig_fs)
    sig_filtered = signal.filtfilt(b, a, sig_filtered)
    sig_filtered = normalize(sig_filtered)
    return sig_filtered


def normalize(x):
    return (x - np.mean(x)) / np.std(x)


def remove_impulse_artifacts(sig):
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


def ppg_pulse_detection(sig, sig_fs, plotflag, fine_search):
    # Linear-phase FIR filter
    ntaps = 3 * sig_fs + 1  # order + 1
    lpd_fp = 7.9
    lpd_fc = 8
    bands = [0, lpd_fp, lpd_fc, 0.5 * sig_fs]
    desired = [1, 0]
    b = signal.remez(ntaps, bands=bands, desired=desired, fs=sig_fs, type='differentiator')
    # w, h = signal.freqz(b, [1], worN=5000, fs=sig_fs)
    # plot_response(w, h, "LPD Filter")
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
    
    if plotflag:
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(np.arange(0, sig.size/sig_fs, 1/sig_fs), sig)
        ax1.plot(n_d_int / sig_fs, sig[n_d_int], 'ro', label='nD')
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(np.arange(0, sig_filtered.size/sig_fs, 1/sig_fs), sig_filtered, label='signal')
        ax2.plot(np.arange(0, sig_filtered.size/sig_fs, 1/sig_fs), threshold, label='threshold')
        ax2.plot(n_d_int / sig_fs, sig_filtered[n_d_int], 'ro', label='nD')
        plt.show()

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
            except InterpolationError:
                pass
    else:
        n_d = np.divide(n_d_int, sig_fs)

    return n_d


def adaptive_thresholding(sig_filt, sig_fs, alfa, refract, tao_rr, thr_incidences):
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


def nfillgap(tk, gaps, current_gap, nfill):
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


def debugplots(ax, dtn, gap, upper_threshold, lower_threshold, nfill, correct):
    ax.cla()
    ax.stem(dtn)
    if correct:
        ax.stem(np.arange(gap, gap+nfill+1), dtn[np.arange(gap, gap+nfill+1)], 'g')
    else:
        ax.stem(np.arange(gap, gap+nfill+1), dtn[np.arange(gap, gap+nfill+1)], 'r')
    ax.set(ylabel='Corrected RR [s]')
    ax.set(xlabel='Samples')
    ax.axhline(upper_threshold, color='k')
    ax.axhline(lower_threshold, color='k')
    plt.pause(0.5)
    plt.show(block=False)


def gap_correction(tk, debug):
    f = []
    ax1 = []
    ax2 = []

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

    if debug:
        f, [ax1, ax2] = plt.subplots(2, 1)

    nfill = 1  # Start filling with one sample
    while gaps:
        # In each iteration, try to fill with one more sample
        for kk in range(0, gaps.size):
            if kk == 0 & debug:
                ax1.cla()
                ax1.stem(dtn)
                ax1.stem(gaps, dtn[gaps], 'r')
                ax1.plot(threshold * kupper, 'k--')
                ax1.set(ylabel='Original RR [s]')

            auxtn = nfillgap(tn, gaps, gaps[kk], nfill)
            auxdtn = np.diff(auxtn)

            correct = np.all(auxdtn[gaps[kk]:(gaps[kk]+nfill+1)] < kupper_fine * threshold_at_gap[kk])
            limit_exceeded = np.any([auxdtn[gaps[kk]:(gaps[kk]+nfill+1)] < klower * threshold_at_gap[kk],
                                     auxdtn[gaps[kk]:(gaps[kk]+nfill+1)] < 0.5])

            if debug:
                if limit_exceeded:
                    debugplots(ax2, auxdtn, gaps[kk], kupper_fine * threshold_at_gap[kk], klower * threshold_at_gap[kk],
                               nfill, False)
                else:
                    debugplots(ax2, auxdtn, gaps[kk], kupper_fine * threshold_at_gap[kk], klower * threshold_at_gap[kk],
                               nfill, correct)

            if limit_exceeded:
                # Check that lower theshold is not exceeded. Use previous nfill instead
                auxtn = nfillgap(tn, gaps, gaps[kk], nfill - 1)
                auxdtn = np.diff(auxtn)
                if debug:
                    debugplots(ax2, auxdtn, gaps[kk], kupper_fine * threshold_at_gap[kk], klower * threshold_at_gap[kk],
                               (nfill - 1), True)
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

    if debug:
        f.clf()

    return tn


def time_metrics(tk):
    rr = np.diff(tk)
    rr[rr == 0] = []  # Remove repeated beats

    threshold = compute_threshold(rr)
    dRR = np.diff(rr)  # (ms)
    rr[rr < 0.7*threshold] = np.nan
    rr[rr > 1.3*threshold] = np.nan

    # Compute time domain indices
    mhr = np.nanmean(60. / rr)  # (beats / min)
    sdnn = 1000 * np.nanstd(rr)  # (ms)
    rmssd = 1000 * np.sqrt(np.sum(np.square(dRR[~np.isnan(dRR)])) / dRR[~np.isnan(dRR)].size)  # (ms)
    sdsd = 1000 * np.nanstd(dRR)  # (ms)
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

def compute_threshold(rr):
    wind = 29
    if rr.size < wind:
        wind = rr.size
        if (wind % 2) != 1:
            wind = wind - 1
    mf = signal.medfilt(np.concatenate((np.flipud(rr[0:wind // 2]), rr, np.flipud(rr[-(wind // 2):])))[:], wind)
    mf[mf > 1.5] = 1.5
    return mf[(wind // 2):-(wind // 2)]


def plot_response(w, h, title):
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(w, np.abs(h))
    ax.set_xlim(0, 10)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain')
    ax.set_title(title)
    ax = fig.add_subplot(212)
    ax.plot(w, np.angle(h))
    ax.set_xlim(0, 10)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase')
    plt.show()


def plot_signal(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y, label='data')
    plt.show()


def save_results_to_storage(results, results_file_path):
    """Save the processing results to Firebase Storage."""
    blob = bucket.blob(results_file_path)

    # Add metadata to the blob. Needed for generate tokens.
    token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": token}
    blob.metadata = metadata

    # Write results to the blob
    json_data = '\n'.join([f"{key}: {value}" for key, value in results.items()])
    blob.upload_from_string(json_data)

    print(f"Results saved at: {results_file_path}")
    

@storage_fn.on_object_finalized(region="europe-west1", memory=MemoryOption.MB_512)
def process_signal(
    event: storage_fn.CloudEvent[storage_fn.StorageObjectData],  
):
    """When a file is uploaded in the Storage bucket, check if PPG and process."""

    file_path = event.data.name

    # Exit if this is triggered on a file that is not a csv file.
    if not file_path.endswith("hz.csv"):
        print(f"Not a PPG file. ({file_path})")
        return
    
    print(f"Processing PPG file. ({file_path})")   

    # Load file
    bucket = storage_client.bucket('bsicos-app.appspot.com')
    file_blob = bucket.get_blob(file_path)
    file_text = file_blob.download_as_text()
    file = open('temp.txt', 'w')  # Create 'temp.txt' file
    file.write(file_text)
    file.close()

    # Generate data matrix and separate arrays
    data_matrix = np.loadtxt(file.name, delimiter=',', skiprows=1)
    try:
        os.remove(file.name)  # Remove 'temp.txt' file
    except Exception as e:
        print(f"Error occurred while removing the temporary file: {str(e)}")
    green = data_matrix[:, 1]
    unixtimestamps = data_matrix[:, 3]

    # Interpolate ppg at fs
    fs = 250
    t_aux = (unixtimestamps - unixtimestamps[0]) / 1000
    t = np.arange(0, t_aux[-1], 1 / fs)
    cs = CubicSpline(t_aux, -green)
    ppg = cs(t)

    # Baseline removal, filtering and normalization of PPG signal
    ppg_filtered = filtering_and_normalization(ppg, fs)
    ppg_filtered = remove_impulse_artifacts(ppg_filtered)

    # Pulse detection
    print("Detecting pulses...")
    ppg_tk = ppg_pulse_detection(ppg_filtered, fs, plotflag=False, fine_search=True)

    # HRV
    print("Computing HRV metrics...")
    td_results = time_metrics(ppg_tk)
    # ppg_tn = gap_correction(ppg_tk, False)

    print("Processing finished. Saving results...")
    results_file_path = os.path.splitext(file_path)[0]
    results_file_path = results_file_path.split('/')[-2:]
    results_file_path = '/'.join(results_file_path)
    results_file_path = f"resultados/{results_file_path}_results.txt"
    save_results_to_storage(td_results, results_file_path)