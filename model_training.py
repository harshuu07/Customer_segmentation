import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import joblib

# Load your dataset
df = pd.read_csv('customer_data.csv')

# Feature engineering (example)
df['TotalSpend'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df['FamilySize'] = df['Kidhome'] + df['Teenhome'] + 1
df['IsParent'] = np.where(df['Kidhome'] + df['Teenhome'] > 0, 1, 0)

# Select features for clustering
features = ['Age', 'Income', 'TotalSpend', 'FamilySize', 'IsParent']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Clustering
cluster = AgglomerativeClustering(n_clusters=4)
labels = cluster.fit_predict(X_pca)

# Save models
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(cluster, 'clustering_model.pkl')
