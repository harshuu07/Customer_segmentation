import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.decomposition import PCA
import os # To check file existence

# --- Page Configuration ---
st.set_page_config(
    page_title="Model Comparison Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# --- Constants ---
DATA_FILE = "data_for_streamlit.csv"
SCALER_FILE = "scaler.pkl"
RF_MODEL_FILE = "random_forest_model.pkl"
KMEANS_MODEL_FILE = "kmeans_model.pkl"
DBSCAN_MODEL_FILE = "dbscan_model.pkl"
TARGET_COLUMN = "Survived" # The column RF predicts

# --- File Check Function ---
def check_files_exist():
    """Checks if all required files are present."""
    files_to_check = [DATA_FILE, SCALER_FILE, RF_MODEL_FILE, KMEANS_MODEL_FILE, DBSCAN_MODEL_FILE]
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    if missing_files:
        st.error(f"Error: The following required files are missing: {', '.join(missing_files)}. "
                 f"Please ensure they are in the same directory as the app.")
        return False
    return True

# --- Load Data and Models (with Caching and Error Handling) ---
@st.cache_resource # Cache resource for models/scaler
def load_pickle(filename):
    """Loads a pickle file with error handling."""
    try:
        with open(filename, 'rb') as f:
            return joblib.load(f)
    except Exception as e:
        st.error(f"Error loading file '{filename}': {e}")
        return None

@st.cache_data # Cache data loading
def load_data(filename):
    """Loads CSV data with error handling."""
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        st.error(f"Error loading data file '{filename}': {e}")
        return None

# --- Main App ---

# 1. Check if essential files exist before proceeding
if not check_files_exist():
    st.stop() # Stop execution if files are missing

# 2. Load Models and Scaler
with st.spinner("Loading models and scaler..."):
    scaler = load_pickle(SCALER_FILE)
    rf_model = load_pickle(RF_MODEL_FILE)
    kmeans_model = load_pickle(KMEANS_MODEL_FILE)
    dbscan_model = load_pickle(DBSCAN_MODEL_FILE)

# Check if loading was successful
if not all([scaler, rf_model, kmeans_model, dbscan_model]):
    st.error("Failed to load one or more model/scaler files. Cannot continue.")
    st.stop()

# 3. Load Data
with st.spinner("Loading data..."):
    data = load_data(DATA_FILE)

if data is None:
    st.error("Failed to load data. Cannot continue.")
    st.stop()

# 4. Prepare Data for Models
try:
    # Separate features (X) and potentially target (y)
    if TARGET_COLUMN in data.columns:
        y = data[TARGET_COLUMN]
        X = data.drop(TARGET_COLUMN, axis=1)
        st.sidebar.success(f"Target column '{TARGET_COLUMN}' found.")
    else:
        y = None
        X = data.copy() # Use all data as features if target is missing
        st.sidebar.warning(f"Target column '{TARGET_COLUMN}' not found in data. "
                           f"Random Forest classification metrics will not be available.")

    # Identify numeric features for scaling (consistent with typical training)
    # Important: Ensure these are the same features the scaler was FIT on!
    # This assumes the scaler was fit on all numeric columns present in X.
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.error("Error: No numeric columns found in the data to apply the scaler.")
        st.stop()

    # Handle potential missing columns (compared to when scaler was fit) - optional but safer
    # Get feature names the scaler expects
    try:
        scaler_features = scaler.feature_names_in_
        # Only scale columns that exist in current data AND were seen by scaler
        cols_to_scale = [col for col in numeric_cols if col in scaler_features]
        if len(cols_to_scale) != len(scaler_features):
             st.warning(f"Warning: Data is missing some columns the scaler was trained on. Scaling only applied to: {cols_to_scale}")
        if not cols_to_scale:
             st.error("Error: None of the data columns match the columns the scaler was trained on.")
             st.stop()
    except AttributeError:
         # Older scalers might not have feature_names_in_
         cols_to_scale = numeric_cols
         st.info("Scaler does not store feature names. Assuming numeric columns match training.")


    # Apply the loaded scaler
    X_scaled_array = scaler.transform(X[cols_to_scale])
    X_scaled = pd.DataFrame(X_scaled_array, columns=cols_to_scale, index=X.index)
    # Note: X_scaled now only contains the scaled numeric columns.
    # If models need other (e.g., non-scaled non-numeric) features, adjust accordingly.

except Exception as e:
    st.error(f"An error occurred during data preparation: {e}")
    st.stop()


# --- UI Layout ---
st.markdown("""
    <style>
        body { font-family: 'Segoe UI', sans-serif; }
        .main { background-color: #f9f9f9; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] { height: 50px; background-color: #e1e1e1; }
        .stTabs [aria-selected="true"] { background-color: #c1c1c1; }
        h1, h2, h3 { color: #3a3a3a; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🔍 Model Explorer")
model_choice = st.sidebar.selectbox(
    "Choose a model to evaluate:",
    ["Random Forest", "KMeans", "DBSCAN"],
    key="model_select"
)

st.title("📊 Machine Learning Model Comparison")
st.write("Dashboard comparing Supervised (Classification) and Unsupervised (Clustering) models.")
st.markdown("---") # Separator

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📈 Metrics", "📉 Visualizations", "🧠 Model Summary"])

# --- Model Specific Display ---

# == Random Forest ==
if model_choice == "Random Forest":
    with st.spinner("Running Random Forest predictions..."):
        try:
            # Use the scaled features the model was trained on
            y_pred = rf_model.predict(X_scaled)
        except Exception as e:
            st.error(f"Error during Random Forest prediction: {e}")
            st.stop()

    with tab1:
        st.subheader("Random Forest: Performance Metrics")
        if y is not None: # Only show metrics if target variable exists
            st.markdown("**Classification Report**")
            try:
                report = classification_report(y, y_pred, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report).transpose())
            except Exception as e:
                st.error(f"Could not generate Classification Report: {e}")

            st.markdown("**Confusion Matrix**")
            try:
                cm = confusion_matrix(y, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, cbar=False)
                ax_cm.set_xlabel("Predicted Label")
                ax_cm.set_ylabel("True Label")
                st.pyplot(fig_cm, clear_figure=True)
            except Exception as e:
                st.error(f"Could not generate Confusion Matrix: {e}")
        else:
            st.warning(f"Target column ('{TARGET_COLUMN}') not found. Cannot display classification metrics.")

    with tab2:
        st.subheader("Random Forest: Feature Importances")
        try:
            importances = rf_model.feature_importances_
            # Ensure features used here match exactly those in X_scaled
            feat_df = pd.DataFrame({
                "Feature": X_scaled.columns, # Use columns from X_scaled
                "Importance": importances
            }).sort_values("Importance", ascending=False)

            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=feat_df.head(15), ax=ax_imp, palette="viridis") # Show top 15
            ax_imp.set_title("Top Feature Importances")
            st.pyplot(fig_imp, clear_figure=True)
        except Exception as e:
            st.error(f"Could not generate Feature Importance plot: {e}")

    with tab3:
        st.subheader("Random Forest: Model Summary")
        st.markdown(f"""
        - **Model Type**: Random Forest Classifier
        - **Learning Type**: Supervised Learning (Classification)
        - **Primary Goal**: Predict a categorical target variable ('{TARGET_COLUMN}' in this case).
        - **Strengths**: Handles non-linear relationships, generally robust to overfitting (with enough trees), provides feature importance measures.
        - **Output**: Predicted class labels and probabilities.
        """)

# == KMeans ==
elif model_choice == "KMeans":
    with st.spinner("Analyzing KMeans clusters..."):
        try:
            cluster_labels = kmeans_model.labels_
            n_clusters = kmeans_model.n_clusters
            # Calculate Silhouette Score only if more than 1 cluster exists
            silhouette = -99 # Default invalid value
            if len(np.unique(cluster_labels)) > 1:
                 silhouette = silhouette_score(X_scaled, cluster_labels)
            else:
                 st.warning("Only one cluster found by KMeans. Silhouette score is not applicable.")

        except Exception as e:
            st.error(f"Error during KMeans analysis: {e}")
            st.stop()

    with tab1:
        st.subheader("KMeans: Clustering Metrics")
        st.metric(label="Number of Clusters (k)", value=n_clusters)
        if silhouette != -99: # Only display if calculated
             st.metric(label="Silhouette Score", value=f"{silhouette:.3f}")
             st.caption("Measures how similar an object is to its own cluster compared to other clusters. Ranges from -1 to 1, higher is better.")

    with tab2:
        st.subheader("KMeans: Cluster Visualization (PCA)")
        try:
            with st.spinner("Reducing dimensions using PCA..."):
                 pca = PCA(n_components=2, random_state=42)
                 reduced_features = pca.fit_transform(X_scaled)

            df_viz = pd.DataFrame(reduced_features, columns=["PC1", "PC2"], index=X.index)
            df_viz["Cluster"] = cluster_labels.astype(str) # Convert labels to string for discrete colors

            fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df_viz, x="PC1", y="PC2", hue="Cluster", palette="tab10", ax=ax_pca, s=50, alpha=0.7)
            ax_pca.set_title(f'KMeans Clusters (k={n_clusters}) Projected onto 2 PCA Components')
            ax_pca.set_xlabel("Principal Component 1")
            ax_pca.set_ylabel("Principal Component 2")
            ax_pca.legend(title="Cluster")
            st.pyplot(fig_pca, clear_figure=True)
        except Exception as e:
            st.error(f"Could not generate PCA visualization: {e}")

    with tab3:
        st.subheader("KMeans: Model Summary")
        st.markdown(f"""
        - **Model Type**: KMeans Clustering
        - **Learning Type**: Unsupervised Learning
        - **Primary Goal**: Partition data points into a predefined number (k={n_clusters}) of distinct, non-overlapping clusters based on feature similarity (distance).
        - **Strengths**: Simple, computationally efficient, scales well to large datasets.
        - **Output**: Cluster assignment label for each data point.
        """)

# == DBSCAN ==
elif model_choice == "DBSCAN":
    with st.spinner("Analyzing DBSCAN clusters..."):
        try:
            cluster_labels = dbscan_model.labels_
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)

            # Calculate Silhouette Score only if valid number of clusters exists
            silhouette = -99 # Default invalid value
            if n_clusters > 1:
                # Exclude noise points for silhouette calculation if desired
                # core_samples_mask = np.zeros_like(dbscan_model.labels_, dtype=bool)
                # core_samples_mask[dbscan_model.core_sample_indices_] = True
                # labels_for_silhouette = cluster_labels[core_samples_mask]
                # data_for_silhouette = X_scaled[core_samples_mask]
                # if len(np.unique(labels_for_silhouette)) > 1:
                #     silhouette = silhouette_score(data_for_silhouette, labels_for_silhouette)

                # Simpler: Calculate on all non-noise points if > 1 cluster
                non_noise_mask = cluster_labels != -1
                if np.sum(non_noise_mask) > 0 and len(np.unique(cluster_labels[non_noise_mask])) > 1:
                     silhouette = silhouette_score(X_scaled[non_noise_mask], cluster_labels[non_noise_mask])
                else:
                     st.warning("Could not calculate Silhouette Score (not enough non-noise points or clusters).")

            elif n_clusters <= 1:
                st.warning("DBSCAN found <= 1 cluster (excluding noise). Silhouette score is not applicable.")

        except Exception as e:
            st.error(f"Error during DBSCAN analysis: {e}")
            st.stop()

    with tab1:
        st.subheader("DBSCAN: Clustering Metrics")
        st.metric(label="Estimated Number of Clusters", value=n_clusters)
        st.metric(label="Number of Noise Points", value=n_noise)
        if silhouette != -99: # Only display if calculated
            st.metric(label="Silhouette Score (non-noise points)", value=f"{silhouette:.3f}")
            st.caption("Measures cluster cohesion vs. separation. Higher is better. Calculated excluding noise points.")


    with tab2:
        st.subheader("DBSCAN: Cluster Visualization (PCA)")
        try:
            with st.spinner("Reducing dimensions using PCA..."):
                 pca = PCA(n_components=2, random_state=42)
                 reduced_features = pca.fit_transform(X_scaled)

            df_viz = pd.DataFrame(reduced_features, columns=["PC1", "PC2"], index=X.index)
            # Convert labels to string. Map -1 (noise) explicitly.
            df_viz["Cluster"] = pd.Series(cluster_labels, index=X.index).map(lambda x: 'Noise (-1)' if x == -1 else f'Cluster {x}').astype(str)

            fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
            # Use a qualitative palette, maybe make noise grey
            unique_labels = df_viz["Cluster"].unique()
            palette = sns.color_palette("tab10", n_colors=len(unique_labels) -1 if 'Noise (-1)' in unique_labels else len(unique_labels))
            color_map = {label: color for label, color in zip(sorted([l for l in unique_labels if l != 'Noise (-1)']), palette)}
            if 'Noise (-1)' in unique_labels:
                color_map['Noise (-1)'] = '#bdbdbd' # Grey for noise

            sns.scatterplot(data=df_viz, x="PC1", y="PC2", hue="Cluster", palette=color_map, ax=ax_pca, s=50, alpha=0.7)
            ax_pca.set_title(f'DBSCAN Clusters Projected onto 2 PCA Components')
            ax_pca.set_xlabel("Principal Component 1")
            ax_pca.set_ylabel("Principal Component 2")
            ax_pca.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig_pca, clear_figure=True)

        except Exception as e:
            st.error(f"Could not generate PCA visualization: {e}")

    with tab3:
        st.subheader("DBSCAN: Model Summary")
        st.markdown(f"""
        - **Model Type**: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
        - **Learning Type**: Unsupervised Learning
        - **Primary Goal**: Group together points that are closely packed together (high-density regions), marking outliers that lie alone in low-density regions as noise. Does not require specifying the number of clusters beforehand.
        - **Strengths**: Can find arbitrarily shaped clusters, robust to noise/outliers.
        - **Output**: Cluster assignment label for each data point (-1 typically indicates noise).
        """)

st.markdown("---")
st.sidebar.info("Dashboard developed to compare ML models.")
