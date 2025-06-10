import os
import sys
# Add project root to path for importing main
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import unittest
from unittest.mock import MagicMock, patch

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
        # Create dummy event with data.name matching test CSV path
        class Data: pass
        class Event: pass
        self.event = Event()
        self.event.data = Data()
        # Use a storage path that ends with hz.csv
        self.event.data.name = 'functions/tests/250610_155821_scppg_30hz.csv'

    @patch('main.storage_client')
    def test_process_signal(self, mock_storage_client):
        # Mock input bucket and blob
        mock_bucket_in = MagicMock()
        mock_blob_in = MagicMock()
        mock_blob_in.download_as_text.return_value = self.csv_text
        mock_bucket_in.get_blob.return_value = mock_blob_in
        mock_storage_client.bucket.return_value = mock_bucket_in
        # Mock global output bucket and blob
        mock_bucket_out = MagicMock()
        main.bucket = mock_bucket_out
        mock_blob_out = MagicMock()
        mock_bucket_out.blob.return_value = mock_blob_out

        # Call the function under test
        main.process_signal(self.event)

        # Check that results are saved to expected path
        expected_out_path = 'resultados/tests/250610_155821_scppg_30hz_results.txt'
        mock_bucket_out.blob.assert_called_once_with(expected_out_path)
        mock_blob_out.upload_from_string.assert_called_once()

        # Clean up temporary file if created
        temp_file = 'temp.txt'
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    unittest.main()