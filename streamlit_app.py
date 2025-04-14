import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="Model Comparison Dashboard", layout="wide", page_icon="📊")

# Load models and data
rf_model = joblib.load("random_forest_model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")
dbscan_model = joblib.load("dbscan_model.pkl")
scaler = joblib.load("scaler.pkl")
data = pd.read_csv("data_for_streamlit.csv")

X = data.drop("Survived", axis=1, errors='ignore')
y = data["Survived"] if "Survived" in data.columns else None
X_scaled = scaler.transform(X)

st.markdown("""
    <style>
        body { font-family: 'Segoe UI', sans-serif; }
        .main { background-color: #f9f9f9; }
        .block-container { padding: 2rem 1rem; }
        h1, h2, h3 { color: #3a3a3a; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🔍 Model Explorer")
model_choice = st.sidebar.selectbox("Choose a model to evaluate:", ["Random Forest", "KMeans", "DBSCAN"])

st.title("📊 Machine Learning Model Comparison")
st.write("A clean dashboard to compare different ML models visually and interactively.")

tab1, tab2, tab3 = st.tabs(["📈 Metrics", "📉 Visuals", "🧠 Model Summary"])

if model_choice == "Random Forest":
    y_pred = rf_model.predict(X_scaled)
    with tab1:
        st.subheader("Classification Report")
        report = classification_report(y, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    with tab2:
        st.subheader("Feature Importances")
        importances = rf_model.feature_importances_
        feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values("Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax)
        st.pyplot(fig)

    with tab3:
        st.markdown("""
        - **Model**: Random Forest Classifier  
        - **Type**: Supervised Learning (Classification)  
        - **Strength**: Handles nonlinear data, robust to overfitting with enough trees  
        - **Output**: Class prediction (e.g., Survived/Not Survived)
        """)

elif model_choice == "KMeans":
    cluster_labels = kmeans_model.labels_
    silhouette = silhouette_score(X_scaled, cluster_labels)
    with tab1:
        st.subheader("Silhouette Score")
        st.metric(label="Score", value=f"{silhouette:.3f}")
    with tab2:
        st.subheader("Cluster Visualization with PCA")
        reduced = PCA(n_components=2).fit_transform(X_scaled)
        df_viz = pd.DataFrame(reduced, columns=["PC1", "PC2"])
        df_viz["Cluster"] = cluster_labels
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_viz, x="PC1", y="PC2", hue="Cluster", palette="tab10", ax=ax)
        st.pyplot(fig)
    with tab3:
        st.markdown("""
        - **Model**: KMeans Clustering  
        - **Type**: Unsupervised Learning  
        - **Strength**: Fast, efficient, scalable  
        - **Output**: Cluster labels based on feature similarity
        """)

elif model_choice == "DBSCAN":
    cluster_labels = dbscan_model.labels_
    silhouette = silhouette_score(X_scaled, cluster_labels)
    with tab1:
        st.subheader("Silhouette Score")
        st.metric(label="Score", value=f"{silhouette:.3f}")
        st.write("**Note**: A score closer to 1 indicates better-defined clusters.")
    with tab2:
        st.subheader("Cluster Visualization with PCA")
        reduced = PCA(n_components=2).fit_transform(X_scaled)
        df_viz = pd.DataFrame(reduced, columns=["PC1", "PC2"])
        df_viz["Cluster"] = cluster_labels
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_viz, x="PC1", y="PC2", hue="Cluster", palette="tab10", ax=ax)
        st.pyplot(fig)
    with tab3:
        st.markdown("""
        - **Model**: DBSCAN (Density-Based Spatial Clustering)  
        - **Type**: Unsupervised Learning  
        - **Strength**: Finds arbitrarily shaped clusters and handles noise  
        - **Output**: Cluster labels (-1 indicates noise)
        """)
