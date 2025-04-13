import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import joblib

# Load the models (if you save them separately, else rebuild below)
# scaler = joblib.load("scaler.pkl")
# pca = joblib.load("pca.pkl")
# cluster_model = joblib.load("cluster_model.pkl")

# For this example, we'll build them again for simplicity (normally you'd reuse the trained models)
def load_model():
    # Simulate training process
    # NOTE: Replace this with actual saved model for real deployment
    # For now, using dummy data shape from training
    scaler = StandardScaler()
    pca = PCA(n_components=3)
    cluster_model = AgglomerativeClustering(n_clusters=4)

    return scaler, pca, cluster_model

scaler, pca, cluster_model = load_model()

st.title("Customer Segmentation - Cluster Prediction")

st.write("Enter customer details below:")

income = st.number_input("Income", min_value=0)
recency = st.slider("Recency (days since last purchase)", 0, 100)
age = st.slider("Age", 18, 100)
spent = st.number_input("Total Amount Spent", min_value=0)
education = st.selectbox("Education Level", ['Undergraduate', 'Graduate', 'Postgraduate'])
living_with = st.selectbox("Living With", ['Alone', 'Partner'])
children = st.slider("Number of Children/Teens at Home", 0, 5)
is_parent = 1 if children > 0 else 0
family_size = (1 if living_with == 'Alone' else 2) + children

# Encode inputs
education_map = {'Undergraduate': 0, 'Graduate': 1, 'Postgraduate': 2}
living_map = {'Alone': 0, 'Partner': 1}

input_data = pd.DataFrame([[
    income, recency, age, spent,
    education_map[education],
    living_map[living_with],
    children, family_size, is_parent
]], columns=[
    'Income', 'Recency', 'Age', 'Spent',
    'Education', 'Living_With', 'Children', 'Family_Size', 'Is_Parent'
])

# Simulate scaling and clustering (in real case, use fitted scaler, pca, and model)
scaled = scaler.fit_transform(input_data)
pca_data = pca.fit_transform(scaled)
cluster = cluster_model.fit_predict(pca_data)

st.success(f"The customer belongs to **Cluster {cluster[0]}**")
