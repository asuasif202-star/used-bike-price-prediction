# Used Bike Price Prediction — Streamlit App

## Project files
- `app.py`: Streamlit web app, preprocessing, model training, metrics, graphs, and prediction form.
- `bikes.csv`: dataset used by the project.
- `requirements.txt`: Python dependencies.

## Run in VS Code (Windows)

1. Extract this ZIP file.
2. Open the extracted folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment (recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run:
   `.venv\Scripts\activate.bat`

5. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

6. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

7. Streamlit opens the app in your browser, usually at `http://localhost:8501`.

## Dataset
The included `bikes.csv` has these columns:
`model_name, model_year, kms_driven, owner, location, mileage, power, price`

You can either use the included CSV or upload another compatible CSV through the sidebar.

## Model details
- Algorithm: RandomForestRegressor
- Train/test split: 80/20
- Metrics: MAE, RMSE, R²
- Categorical columns are one-hot encoded.
- Missing numeric values use median imputation; missing categorical values use a `Missing` category.

## Notes
The model retrains when the app starts. Prediction results are estimates based on the supplied listings, not guaranteed sale prices. The bike age reference year is set to 2026 in `app.py`; update `REFERENCE_YEAR` if needed.
