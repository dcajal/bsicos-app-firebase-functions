"""Firebase Cloud Function entry point for PPG signal processing."""

import firebase_admin
from firebase_functions import storage_fn
from firebase_admin import credentials
from firebase_functions.options import MemoryOption

from ppg_processor import process_ppg_file, save_processing_results

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'bsicos-app.appspot.com'})
    

@storage_fn.on_object_finalized(region="europe-west1", memory=MemoryOption.MB_512)
def process_signal(
    event: storage_fn.CloudEvent[storage_fn.StorageObjectData],   # type: ignore
):
    """When a file is uploaded in the Storage bucket, check if PPG and process."""

    file_path = event.data.name

    # Exit if this is triggered on a file that is not a csv file.
    if not file_path.endswith("hz.csv"):
        print(f"Not a PPG file. ({file_path})")
        return
    
    try:
        # Process the PPG file
        results = process_ppg_file(file_path)
        
        # Save results to storage
        save_processing_results(results, file_path)
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return

