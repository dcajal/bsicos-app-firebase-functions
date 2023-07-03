import pathlib
import numpy as np

from firebase_functions import storage_fn
from firebase_admin import initialize_app
from scipy.interpolate import CubicSpline
# from lib.delineation import ppg_pulse_detection
# from lib.filters import filtering_and_normalization, remove_impulse_artifacts
# from lib.hrv import time_metrics

initialize_app()


@storage_fn.on_object_finalized(region="europe-west1")
def process_signal(
    event: storage_fn.CloudEvent[storage_fn.StorageObjectData],  
):
    """When a file is uploaded in the Storage bucket, check if PPG and process."""

    file_path = pathlib.PurePath(event.data.name)

    # Exit if this is triggered on a file that is not a csv file.
    if not file_path.endswith("hz.csv"):
        print(f"Not a PPG file. ({file_path})")
        return
    
    print(f"Processing PPG file. ({file_path})")   
    fs = 250

    # Load file
    my_file = np.genfromtxt(file_path, delimiter=',')
    data_matrix = np.delete(my_file, 0, 0)
    green = data_matrix[:, 1]
    unixtimestamps = data_matrix[:, 3]

    # Interpolate ppg at fs
    t_aux = (unixtimestamps - unixtimestamps[0]) / 1000
    t = np.arange(0, t_aux[-1], 1 / fs)
    cs = CubicSpline(t_aux, -green)
    ppg = cs(t)

    # Baseline removal, filtering and normalization of PPG signal
    # ppg_filtered = filtering_and_normalization(ppg, fs)
    # ppg_filtered = remove_impulse_artifacts(ppg_filtered)

    # Pulse detection
    # ppg_tk = ppg_pulse_detection(ppg_filtered, fs, plotflag=False)

    # HRV
    # time_metrics(ppg_tk)
    # ppg_tn = gap_correction(ppg_tk, True)
