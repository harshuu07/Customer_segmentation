
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# --- Load your dataset ---
df = pd.read_csv("your_dataset.csv")  # Replace with your actual file name

# --- Feature + Target ---
X = df.drop("Survived", axis=1)  # Update target name if different
y = df["Survived"]

# --- Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train Random Forest Classifier ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
joblib.dump(rf, "random_forest_model.pkl")

# --- Train KMeans ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
joblib.dump(kmeans, "kmeans_model.pkl")

# --- Train DBSCAN ---
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan.fit(X_scaled)
joblib.dump(dbscan, "dbscan_model.pkl")

# --- Save Scaler and Data for Streamlit ---
joblib.dump(scaler, "scaler.pkl")
df.to_csv("data_for_streamlit.csv", index=False)

print("✅ All models and data saved successfully!")
