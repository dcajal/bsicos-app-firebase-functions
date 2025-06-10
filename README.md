# Firebase functions for BSICoS App

Methods for extracting pulse rate variability metrics from smartphone photoplethysmography. The execution is automatic when a file is uploaded to the cloud.

## Development Setup

1. Create a virtual environment:

   ```bash
   cd functions
   python3.13 -m venv venv
   source venv/bin/activate
   ```

2. Install production and development dependencies:

   ```bash
   pip install -r requirements.txt
   pip install -r dev-requirements.txt
   ```

3. Run tests:

   ```bash
   cd functions
   source venv/bin/activate
   pytest tests
   ```

## Deploying to Firebase

1. Generate production requirements file and ensure only production dependencies are listed:

   ```bash
   pip freeze > functions/requirements.txt
   ```

2. Deploy the code to Firebase:

   ```bash
   firebase deploy --only functions
   ```

3. Verify the deployment in the Firebase Console.
