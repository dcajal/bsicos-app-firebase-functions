"""Storage utilities for Firebase Cloud Storage."""

from google.cloud import storage as gcs
from uuid import uuid4
from typing import Dict


def save_results_to_storage(results: Dict[str, int], results_file_path: str) -> None:
    """Save the processing results to Firebase Storage."""
    storage_client = gcs.Client()
    bucket = storage_client.bucket('bsicos-app.appspot.com')
    blob = bucket.blob(results_file_path)

    # Add metadata to the blob. Needed for generate tokens.
    token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": token}
    blob.metadata = metadata

    # Write results to the blob
    json_data = '\n'.join([f"{key}: {value}" for key, value in results.items()])
    blob.upload_from_string(json_data)

    print(f"Results saved at: {results_file_path}")


def load_file_from_storage(file_path: str, bucket_name: str = 'bsicos-app.appspot.com') -> str:
    """Load file content from Firebase Storage."""
    storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)
    file_blob = bucket.get_blob(file_path)
    
    if file_blob is None:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return file_blob.download_as_text()
