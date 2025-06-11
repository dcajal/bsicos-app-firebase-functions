import os
import sys
# Add project root to path for importing main
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import unittest
from unittest.mock import MagicMock, patch
from typing import Any, cast

import firebase_admin
# Mock firebase_admin initialization to avoid reading serviceAccountKey.json
firebase_admin.credentials.Certificate = MagicMock(return_value=None)
firebase_admin.initialize_app = MagicMock(return_value=None)

# Prevent instantiation of real storage client to avoid ADC error
import google.cloud.storage as gcs
gcs.Client = MagicMock(return_value=MagicMock())

import os, json
# Provide FIREBASE_CONFIG for storage fn decorator
os.environ['FIREBASE_CONFIG'] = json.dumps({'storageBucket': 'bsicos-app.appspot.com'})

# Stub the storage_fn decorator to a no-op
import firebase_functions.storage_fn as storage_fn
storage_fn.on_object_finalized = lambda *args, **kwargs: (lambda f: f)

import main
import pandas as pd

class TestMainFunctionality(unittest.TestCase):
    def setUp(self):
        # Read CSV text for input blob
        csv_path = os.path.join(os.path.dirname(__file__), '250610_155821_scppg_30hz.csv')
        with open(csv_path, 'r') as f:
            self.csv_text = f.read()
        
        # Create dummy event with proper typing to match CloudEvent[StorageObjectData]
        class MockStorageObjectData:
            def __init__(self, name: str):
                self.name = name
        
        class MockCloudEvent:
            def __init__(self, data: MockStorageObjectData):
                self.data = data
        
        # Use a storage path that ends with hz.csv
        self.event = MockCloudEvent(MockStorageObjectData('functions/tests/250610_155821_scppg_30hz.csv'))

    @patch('main.gcs.Client')
    def test_process_signal(self, mock_gcs_client):
        # Mock storage client and buckets
        mock_storage_client = MagicMock()
        mock_gcs_client.return_value = mock_storage_client
        
        # Mock input bucket and blob
        mock_bucket = MagicMock()
        mock_blob_in = MagicMock()
        mock_blob_in.download_as_text.return_value = self.csv_text
        mock_bucket.get_blob.return_value = mock_blob_in
        
        # Mock output blob for saving results
        mock_blob_out = MagicMock()
        mock_bucket.blob.return_value = mock_blob_out
        
        # Configure storage client to return the same bucket for both operations
        mock_storage_client.bucket.return_value = mock_bucket

        # Call the function under test
        # Cast to Any to bypass type checking for this test mock
        event_typed = cast(Any, self.event)
        main.process_signal(event_typed)

        # Check that results are saved to expected path
        expected_out_path = 'resultados/tests/250610_155821_scppg_30hz_results.txt'
        mock_bucket.blob.assert_called_with(expected_out_path)
        mock_blob_out.upload_from_string.assert_called_once()

        # Clean up temporary file if created
        temp_file = 'temp.txt'
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    unittest.main()