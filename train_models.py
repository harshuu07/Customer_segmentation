import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Optional: Good practice, but final models often trained on all data
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import classification_report, silhouette_score
import joblib
import warnings

warnings.filterwarnings('ignore', category=FutureWarning) # Suppress KMeans n_init warning if using older sklearn

# --- Configuration ---
DATA_FILE = "data_for_streamlit.csv"
TARGET_COLUMN = "Survived" # Make sure this column exists in your CSV for Random Forest

# --- Hyperparameters (Consider Tuning These!) ---
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42

KMEANS_N_CLUSTERS = 3 # Example: Choose based on your data/elbow method
KMEANS_RANDOM_STATE = 42

DBSCAN_EPS = 0.5     # Example: Very sensitive, tune carefully
DBSCAN_MIN_SAMPLES = 5 # Example: Tune carefully

# --- 1. Load Data ---
print(f"Loading data from {DATA_FILE}...")
try:
    data = pd.read_csv(DATA_FILE)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Data file '{DATA_FILE}' not found. Please place it in the same directory.")
    exit()

# --- 2. Prepare Data ---
print("Preparing data...")
if TARGET_COLUMN not in data.columns:
    print(f"ERROR: Target column '{TARGET_COLUMN}' not found in the data. Cannot train Random Forest.")
    # Decide if you want to exit or proceed only with clustering
    # For now, we'll exit if the target is missing for RF.
    exit() # Exit if RF target is missing

# Separate features (X) and target (y)
X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]

# Identify numeric columns for imputation and scaling
# (Assuming non-numeric columns are IDs or categorical features already handled)
# If you have categorical features, they need encoding BEFORE scaling.
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
print(f"Identified numeric columns for scaling: {numeric_cols}")

# Handle Missing Values (Simple Median Imputation for Numeric Features)
# IMPORTANT: Review if this is the right strategy for your data!
if X[numeric_cols].isnull().sum().sum() > 0:
    print("Handling missing values using median imputation...")
    for col in numeric_cols:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
    print("Missing values handled.")
else:
    print("No missing numeric values found.")

# Ensure all features are numeric for scaling (handle potential errors)
try:
    X_numeric = X[numeric_cols].astype(float)
except ValueError as e:
    print(f"ERROR: Could not convert all selected columns to numeric. Check your data. Details: {e}")
    exit()

# --- 3. Scale Features ---
print("Scaling features using StandardScaler...")
scaler = StandardScaler()
# Fit the scaler on the numeric features ONLY and transform them
X_scaled = scaler.fit_transform(X_numeric)
# Convert scaled data back to DataFrame for consistency (optional, but can be helpful)
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols, index=X.index)
# If you had non-numeric columns you wanted to keep, you'd merge them back here.
# For this example, we assume all columns processed were numeric for simplicity matching Streamlit app.
print("Features scaled.")

# --- 4. Train and Save Scaler ---
print("Saving the scaler...")
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved to scaler.pkl")

# --- 5. Train and Save Random Forest ---
print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    random_state=RF_RANDOM_STATE,
    # Add other parameters if needed (e.g., max_depth, min_samples_leaf)
)
# Train on the SCALED numeric data
rf_model.fit(X_scaled_df, y)
print("Random Forest training complete.")
print("Saving Random Forest model...")
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Model saved to random_forest_model.pkl")

# Optional: Evaluate RF on training data (just for a quick check)
# y_pred_rf = rf_model.predict(X_scaled_df)
# print("\n--- Random Forest Training Classification Report ---")
# print(classification_report(y, y_pred_rf))
# print("----------------------------------------------------")


# --- 6. Train and Save KMeans ---
print(f"Training KMeans model (k={KMEANS_N_CLUSTERS})...")
kmeans_model = KMeans(
    n_clusters=KMEANS_N_CLUSTERS,
    random_state=KMEANS_RANDOM_STATE,
    n_init='auto' # Explicitly set n_init for future versions
)
# Fit on the SCALED numeric data
kmeans_model.fit(X_scaled_df)
print("KMeans training complete.")
print("Saving KMeans model...")
joblib.dump(kmeans_model, 'kmeans_model.pkl')
print("Model saved to kmeans_model.pkl")

# Optional: Evaluate KMeans on training data
# kmeans_labels = kmeans_model.labels_
# try:
#     silhouette_avg_kmeans = silhouette_score(X_scaled_df, kmeans_labels)
#     print(f"\n--- KMeans Training Silhouette Score: {silhouette_avg_kmeans:.3f} ---")
# except ValueError:
#      print("\n--- KMeans Training: Could not calculate Silhouette Score (likely only 1 cluster found). ---")
# print("----------------------------------------------------")


# --- 7. Train and Save DBSCAN ---
print(f"Training DBSCAN model (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
# Note: DBSCAN is sensitive to parameters and scaling. eps=0.5 is common for scaled data but needs tuning.
dbscan_model = DBSCAN(
    eps=DBSCAN_EPS,
    min_samples=DBSCAN_MIN_SAMPLES
)
# Fit on the SCALED numeric data
dbscan_model.fit(X_scaled_df) # DBSCAN doesn't predict, it fits and assigns labels_
print("DBSCAN training complete.")
print("Saving DBSCAN model...")
joblib.dump(dbscan_model, 'dbscan_model.pkl')
print("Model saved to dbscan_model.pkl")

# Optional: Evaluate DBSCAN on training data
# dbscan_labels = dbscan_model.labels_
# n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
# n_noise_db = list(dbscan_labels).count(-1)
# print(f"\n--- DBSCAN Training Results ---")
# print(f"Estimated number of clusters: {n_clusters_db}")
# print(f"Estimated number of noise points: {n_noise_db}")
# if n_clusters_db > 1: # Silhouette score requires at least 2 clusters
#     try:
#         silhouette_avg_dbscan = silhouette_score(X_scaled_df, dbscan_labels)
#         print(f"Silhouette Score: {silhouette_avg_dbscan:.3f}")
#     except ValueError:
#         print("Could not calculate Silhouette Score (issue with labels/data).")
# else:
#     print("Silhouette Score cannot be calculated (less than 2 clusters found).")
# print("-----------------------------")


print("\n--- All models and scaler have been trained and saved successfully! ---")
print("You should now have:")
print("- scaler.pkl")
print("- random_forest_model.pkl")
print("- kmeans_model.pkl")
print("- dbscan_model.pkl")
print("Place these files in the same directory as your Streamlit app.py and data_for_streamlit.csv.")
