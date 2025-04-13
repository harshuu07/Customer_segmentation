import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load pre-trained models
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
cluster_model = joblib.load('clustering_model.pkl')

st.title("Customer Segmentation App")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=0, value=50000)
mnt_wines = st.number_input("Wine Spend", min_value=0, value=100)
mnt_fruits = st.number_input("Fruit Spend", min_value=0, value=50)
mnt_meat = st.number_input("Meat Spend", min_value=0, value=150)
mnt_fish = st.number_input("Fish Spend", min_value=0, value=70)
mnt_sweets = st.number_input("Sweet Spend", min_value=0, value=30)
mnt_gold = st.number_input("Gold Spend", min_value=0, value=20)
kidhome = st.number_input("Number of Kids at Home", min_value=0, max_value=10, value=0)
teenhome = st.number_input("Number of Teens at Home", min_value=0, max_value=10, value=0)

# Feature engineering
total_spend = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweets + mnt_gold
family_size = kidhome + teenhome + 1
is_parent = 1 if (kidhome + teenhome) > 0 else 0

# Prepare input data
input_data = pd.DataFrame([[age, income, total_spend, family_size, is_parent]],
                          columns=['Age', 'Income', 'TotalSpend', 'FamilySize', 'IsParent'])

# Scale and transform input data
input_scaled = scaler.transform(input_data)
input_pca = pca.transform(input_scaled)

# Predict cluster
cluster_label = cluster_model.fit_predict(input_pca)[0]

st.write(f"Predicted Customer Segment: **Cluster {cluster_label}**")
