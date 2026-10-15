import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Used Bike Price Prediction", page_icon="🏍️", layout="wide")

FEATURES = [
    "model_name", "model_year", "kms_driven", "owner",
    "location", "mileage", "power", "bike_age"
]
TARGET = "price"
REFERENCE_YEAR = 2026

@st.cache_data
def load_and_clean_data(uploaded_file=None):
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    else:
        local_path = os.path.join(os.path.dirname(__file__), "bikes.csv")
        if not os.path.exists(local_path):
            return None, None
        raw = pd.read_csv(local_path)

    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    required = ["model_name", "model_year", "kms_driven", "owner", "location", "mileage", "power", "price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, f"Missing required columns: {', '.join(missing)}"

    def numeric_from_text(series):
        s = series.astype("string").str.replace(",", "", regex=False)
        extracted = s.str.extract(r"([-+]?(?:\d*\.?\d+))", expand=False)
        return pd.to_numeric(extracted, errors="coerce")

    for col in ["model_year", "kms_driven", "mileage", "power", "price"]:
        df[col] = numeric_from_text(df[col])

    for col in ["model_name", "owner", "location"]:
        df[col] = df[col].astype("string").str.strip().astype(object)
        df[col] = df[col].where(pd.notna(df[col]), np.nan)
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    original_rows = len(df)
    df = df.drop_duplicates()
    df = df[df["price"].notna() & (df["price"] > 0)].copy()
    df.loc[~df["model_year"].between(1950, REFERENCE_YEAR), "model_year"] = np.nan
    df["bike_age"] = REFERENCE_YEAR - df["model_year"]
    if df.empty:
        return None, "No usable rows remain after cleaning. Check the price column and dataset."
    return (df, original_rows), None

@st.cache_resource
def train_model(data):
    X = data[FEATURES].copy()
    y = data[TARGET].astype(float).copy()

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    for col in categorical_features:
        X[col] = X[col].astype(object)
        X[col] = X[col].where(pd.notna(X[col]), np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions)),
    }
    return model, metrics, y_test, predictions

st.title("🏍️ Used Bike Price Prediction")
st.caption("Estimate a used bike's resale price using a Random Forest Regression model.")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload bikes.csv (optional)", type=["csv"])
    st.caption("If no file is uploaded, the app reads bikes.csv from the project folder.")
    st.divider()
    st.write("**Model:** Random Forest Regressor")
    st.write("**Test split:** 80% training / 20% testing")

loaded, error = load_and_clean_data(uploaded)
if error:
    st.error(error)
    st.info("Expected columns: model_name, model_year, kms_driven, owner, location, mileage, power, price.")
    st.stop()
if loaded is None:
    st.warning("Place bikes.csv beside app.py, or upload it using the sidebar.")
    st.stop()

df, original_rows = loaded
model, metrics, y_test, predictions = train_model(df)

tab_predict, tab_analysis, tab_data = st.tabs(["🔮 Predict Price", "📊 Model Analysis", "🗂️ Dataset"])

with tab_predict:
    st.subheader("Enter bike details")
    col1, col2 = st.columns(2)
    model_options = sorted(df["model_name"].dropna().astype(str).unique().tolist())
    owner_options = sorted(df["owner"].dropna().astype(str).unique().tolist())
    location_options = sorted(df["location"].dropna().astype(str).unique().tolist())

    with st.form("prediction_form"):
        with col1:
            selected_model = st.selectbox("Bike model", model_options)
            model_year = st.number_input("Manufacturing year", min_value=1950, max_value=REFERENCE_YEAR, value=2020, step=1)
            kms_driven = st.number_input("Kilometres driven", min_value=0, max_value=1_000_000, value=20000, step=1000)
            owner = st.selectbox("Owner type", owner_options)
        with col2:
            location = st.selectbox("Location", location_options)
            mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=250.0, value=40.0, step=1.0)
            power = st.number_input("Power (bhp)", min_value=0.0, max_value=500.0, value=15.0, step=1.0)
        submitted = st.form_submit_button("Predict bike price", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "model_name": selected_model,
            "model_year": float(model_year),
            "kms_driven": float(kms_driven),
            "owner": owner,
            "location": location,
            "mileage": float(mileage),
            "power": float(power),
            "bike_age": float(REFERENCE_YEAR - model_year),
        }], columns=FEATURES)
        estimated_price = float(model.predict(row)[0])
        st.success(f"Estimated used bike price: ₹{estimated_price:,.0f}")
        st.caption("This is a machine-learning estimate, not a guaranteed selling price.")

with tab_analysis:
    st.subheader("Evaluation on the held-out test set")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"₹{metrics['MAE']:,.0f}")
    c2.metric("RMSE", f"₹{metrics['RMSE']:,.0f}")
    c3.metric("R² score", f"{metrics['R2']:.3f}")
    st.caption("MAE/RMSE are in price units. Higher R² generally indicates a better fit; results depend on the dataset.")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ax.hist(df["price"].dropna(), bins=35)
        ax.set_title("Distribution of Used Bike Prices")
        ax.set_xlabel("Price (INR)")
        ax.set_ylabel("Number of listings")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.5, s=16)
        lower = min(float(np.min(y_test)), float(np.min(predictions)))
        upper = max(float(np.max(y_test)), float(np.max(predictions)))
        ax.plot([lower, upper], [lower, upper], linestyle="--")
        ax.set_title("Actual vs Predicted Price")
        ax.set_xlabel("Actual price (INR)")
        ax.set_ylabel("Predicted price (INR)")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Short analysis:** Predictions close to the diagonal are more accurate. MAE is the average absolute error, while RMSE penalizes large errors more strongly. Rare bike models and unusually high/low prices may be harder to predict.")

with tab_data:
    st.subheader("Dataset overview")
    a, b, c = st.columns(3)
    a.metric("Rows in uploaded file", f"{original_rows:,}")
    b.metric("Rows after cleaning", f"{len(df):,}")
    c.metric("Number of columns", f"{len(df.columns):,}")
    st.dataframe(df.head(100), use_container_width=True)
    st.write("**Price summary**")
    st.dataframe(df["price"].describe().to_frame().T, use_container_width=True)
    st.download_button(
        "Download cleaned dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_bikes.csv",
        mime="text/csv"
    )

with st.sidebar:
    st.divider()
    if st.button("Save trained model"):
        model_path = os.path.join(os.path.dirname(__file__), "used_bike_price_model.joblib")
        joblib.dump(model, model_path)
        st.success(f"Saved: {os.path.basename(model_path)}")
