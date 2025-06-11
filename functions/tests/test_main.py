import os
import sys
import json
import unittest
from typing import Any, cast
from unittest.mock import MagicMock, patch

# Add project root to path for importing main module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Mock Firebase Admin to avoid credential requirements during testing
import firebase_admin
firebase_admin.credentials.Certificate = MagicMock(return_value=None)
firebase_admin.initialize_app = MagicMock(return_value=None)

# Mock Google Cloud Storage to prevent authentication errors
import google.cloud.storage as gcs
gcs.Client = MagicMock(return_value=MagicMock())

# Mock Firebase Functions environment
os.environ['FIREBASE_CONFIG'] = json.dumps({'storageBucket': 'bsicos-app.appspot.com'})
import firebase_functions.storage_fn as storage_fn
storage_fn.on_object_finalized = lambda *args, **kwargs: (lambda f: f)

import main

class MockStorageObjectData:
    """Mock object representing Firebase Storage object data."""
    def __init__(self, name: str):
        self.name = name


class MockCloudEvent:
    """Mock object representing Firebase CloudEvent."""
    def __init__(self, data: MockStorageObjectData):
        self.data = data


class TestMainFunctionality(unittest.TestCase):
    """Test suite for main signal processing functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self._load_test_csv()
        self._create_mock_event()
    
    def _load_test_csv(self):
        """Load test CSV data from file."""
        csv_path = os.path.join(os.path.dirname(__file__), '250610_155821_scppg_30hz.csv')
        with open(csv_path, 'r') as f:
            self.csv_text = f.read()
    
    def _create_mock_event(self):
        """Create mock Firebase CloudEvent for testing."""
        file_path = 'functions/tests/250610_155821_scppg_30hz.csv'
        mock_data = MockStorageObjectData(file_path)
        self.event = MockCloudEvent(mock_data)

    @patch('main.gcs.Client')
    def test_process_signal(self, mock_gcs_client):
        """Test the main signal processing function with mocked storage."""
        self._setup_storage_mocks(mock_gcs_client)
        self._execute_signal_processing()
        self._verify_results_saved()
        self._cleanup_temp_files()
    
    def _setup_storage_mocks(self, mock_gcs_client):
        """Configure all storage-related mocks."""
        # Create mock storage client and bucket
        self.mock_storage_client = MagicMock()
        mock_gcs_client.return_value = self.mock_storage_client
        
        self.mock_bucket = MagicMock()
        self.mock_storage_client.bucket.return_value = self.mock_bucket
        
        # Mock input blob (file download)
        mock_blob_in = MagicMock()
        mock_blob_in.download_as_text.return_value = self.csv_text
        self.mock_bucket.get_blob.return_value = mock_blob_in
        
        # Mock output blob (results upload)
        self.mock_blob_out = MagicMock()
        self.mock_bucket.blob.return_value = self.mock_blob_out
    
    def _execute_signal_processing(self):
        """Execute the main signal processing function."""
        event_typed = cast(Any, self.event)
        main.process_signal(event_typed)
    
    def _verify_results_saved(self):
        """Verify that results were saved to the expected storage path."""
        expected_path = 'resultados/tests/250610_155821_scppg_30hz_results.txt'
        self.mock_bucket.blob.assert_called_with(expected_path)
        self.mock_blob_out.upload_from_string.assert_called_once()
    
    def _cleanup_temp_files(self):
        """Clean up any temporary files created during testing."""
        temp_file = 'temp.txt'
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    unittest.main()