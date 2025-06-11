# Firebase functions for BSICoS App

Methods for extracting pulse rate variability metrics from smartphone photoplethysmography. The execution is automatic when a file is uploaded to the cloud.

## Code Architecture

The codebase has been refactored into a modular structure for better maintainability and testability:

### Project Structure

```
functions/
├── main.py                 # Firebase Cloud Function entry point
├── signal_processing.py    # Signal processing utilities
├── hrv_analysis.py        # HRV analysis functions
├── storage_utils.py       # Firebase Storage operations
├── ppg_processor.py       # Main PPG processing pipeline
├── gap_correction.py      # Optional gap correction functionality
└── tests/
    └── test_main.py       # Unit tests
```

### Module Descriptions

- **`main.py`**: Firebase Cloud Function entry point that triggers when files are uploaded
- **`signal_processing.py`**: Core algorithms for filtering, normalization, artifact removal, and pulse detection
- **`hrv_analysis.py`**: Heart Rate Variability analysis and time-domain metrics computation
- **`storage_utils.py`**: Firebase Cloud Storage operations for loading and saving files
- **`ppg_processor.py`**: Main processing pipeline that orchestrates the complete workflow
- **`gap_correction.py`**: Optional advanced gap correction functionality (preserved for future use)

### Processing Pipeline

1. **File Loading**: Load CSV data from Firebase Storage
2. **Data Preprocessing**: Convert timestamps and extract PPG signal
3. **Signal Processing**:
   - Interpolate to 250 Hz sampling rate
   - Apply filtering and normalization
   - Remove impulse artifacts
4. **Pulse Detection**: Detect heartbeat pulses using adaptive thresholding
5. **HRV Analysis**: Compute time-domain HRV metrics (MHR, SDNN, RMSSD, SDSD, pNN50)
6. **Results Storage**: Save metrics to Firebase Storage with download tokens

## Development Setup

1. Create a virtual environment:

   ```bash
   cd functions
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install production and development dependencies:

   ```bash
   pip install -r requirements.txt
   pip install -r dev-requirements.txt
   ```

3. Run tests:

   ```bash
   pytest tests
   ```

## Deploying to Firebase

1. Download and configure the Firebase service account key:

   - Go to the [Firebase Console](https://console.firebase.google.com/)
   - Select your project
   - Go to **Project Settings** (gear icon) → **Service accounts**
   - Click **Generate new private key**
   - Download the JSON file and rename it to `serviceAccountKey.json`
   - Place it in the functions directory: `functions/serviceAccountKey.json`

   **Note**: This file contains sensitive credentials and should never be committed to version control.

2. Generate production requirements file and ensure only production dependencies are listed:

   ```bash
   pip freeze > functions/requirements.txt
   ```

3. Deploy the code to Firebase:

   ```bash
   firebase deploy --only functions
   ```

4. Verify the deployment in the Firebase Console.
