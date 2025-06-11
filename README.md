# Firebase functions for BSICoS App

Methods for extracting pulse rate variability metrics from smartphone photoplethysmography. The execution is automatic when a file is uploaded to the cloud.

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
