import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib

# Load your data
df = pd.read_csv("your_dataset.csv")  # Replace with your CSV

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
joblib.dump(rf, "random_forest_model.pkl")

# Train KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
joblib.dump(kmeans, "kmeans_model.pkl")

# Train DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan.fit(X_scaled)
joblib.dump(dbscan, "dbscan_model.pkl")

# Save the scaler and data
joblib.dump(scaler, "scaler.pkl")
df.to_csv("data_for_streamlit.csv", index=False)

print("✅ Models and data saved.")
