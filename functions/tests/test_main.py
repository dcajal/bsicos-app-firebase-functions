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

    @patch('ppg_processor.load_file_from_storage')
    @patch('ppg_processor.save_results_to_storage')
    def test_process_signal(self, mock_save_results, mock_load_file):
        """Test the main signal processing function with mocked storage."""
        self._setup_storage_mocks(mock_load_file, mock_save_results)
        self._execute_signal_processing()
        self._verify_results_saved(mock_save_results)
        self._cleanup_temp_files()
    
    def _setup_storage_mocks(self, mock_load_file, mock_save_results):
        """Configure all storage-related mocks."""
        # Mock file loading to return test CSV content
        mock_load_file.return_value = self.csv_text
        
        # Mock save results (no return value needed)
        mock_save_results.return_value = None
    
    def _execute_signal_processing(self):
        """Execute the main signal processing function."""
        event_typed = cast(Any, self.event)
        main.process_signal(event_typed)
    
    def _verify_results_saved(self, mock_save_results):
        """Verify that results were saved to the expected storage path."""
        # Verify that save_results_to_storage was called
        mock_save_results.assert_called_once()
        
        # Get the arguments passed to save_results_to_storage
        call_args = mock_save_results.call_args
        results_dict = call_args[0][0]  # First argument (results dictionary)
        file_path = call_args[0][1]     # Second argument (file path)
        
        # Verify the results dictionary contains expected keys
        expected_keys = ["MHR [beats/min]", "SDNN [ms]", "RMSSD [ms]", "SDSD [ms]", "pNN50"]
        for key in expected_keys:
            self.assertIn(key, results_dict)
            self.assertIsInstance(results_dict[key], int)
        
        # Verify the file path is correct
        expected_path = 'resultados/tests/250610_155821_scppg_30hz_results.txt'
        self.assertEqual(file_path, expected_path)
    
    def _cleanup_temp_files(self):
        """Clean up any temporary files created during testing."""
        temp_file = 'temp.txt'
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    unittest.main()