"""PPG processing pipeline."""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from typing import Dict

from signal_processing import filtering_and_normalization, remove_impulse_artifacts, ppg_pulse_detection
from hrv_analysis import time_metrics
from storage_utils import load_file_from_storage, save_results_to_storage


def process_ppg_file(file_path: str) -> Dict[str, int]:
    """Process a PPG file and return HRV metrics."""
    print(f"Processing PPG file. ({file_path})")
    
    # Load file from storage
    try:
        file_text = load_file_from_storage(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    
    # Create temporary file
    temp_file = 'temp.txt'
    with open(temp_file, 'w') as f:
        f.write(file_text)

    try:
        # Generate data matrix
        data_matrix = pd.read_csv(temp_file, sep=',')
        
        # Convert 'Unix timestamps' to datetime and milliseconds
        ts_dt = pd.to_datetime(data_matrix['Unix timestamps'])
        unixtimestamps = np.array(pd.to_numeric(ts_dt)) // 1_000_000
        red = data_matrix['Red'].to_numpy()

        # Interpolate ppg at fs
        fs = 250
        t_aux = (unixtimestamps - unixtimestamps[0]) / 1000
        t = np.arange(0, t_aux[-1], 1 / fs)
        cs = CubicSpline(t_aux, -red)
        ppg = cs(t)

        # Baseline removal, filtering and normalization of PPG signal
        ppg_filtered = filtering_and_normalization(ppg, fs)
        ppg_filtered = remove_impulse_artifacts(ppg_filtered)

        # Pulse detection
        print("Detecting pulses...")
        ppg_tk = ppg_pulse_detection(ppg_filtered, fs, fine_search=True)

        # HRV
        print("Computing HRV metrics...")
        td_results = time_metrics(ppg_tk)
        
        return td_results
        
    except Exception as e:
        print(f"Error occurred while processing the CSV file: {str(e)}")
        raise
    finally:
        # Clean up temporary file
        try:
            os.remove(temp_file)
        except Exception as e:
            print(f"Error occurred while removing the temporary file: {str(e)}")


def save_processing_results(results: Dict[str, int], original_file_path: str) -> None:
    """Save processing results to storage."""
    print("Processing finished. Saving results...")
    
    # Generate results file path
    results_file_path = os.path.splitext(original_file_path)[0]
    results_file_path = results_file_path.split('/')[-2:]
    results_file_path = '/'.join(results_file_path)
    results_file_path = f"resultados/{results_file_path}_results.txt"
    
    save_results_to_storage(results, results_file_path)
