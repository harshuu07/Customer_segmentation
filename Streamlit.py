import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample coefficients and odds ratios
data = {
    'Feature': [
        'Pclass', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare',
        'Cabin_B', 'Title_Master', 'Title_Miss', 'Title_Mr',
        'Title_Mrs', 'Title_Rev', 'Title_Rare',
        'First_Letter_Cabin_A', 'First_Letter_Cabin_B', 'First_Letter_Cabin_C',
        'First_Letter_Cabin_D', 'First_Letter_Cabin_E', 'First_Letter_Cabin_F',
        'First_Letter_Cabin_G', 'First_Letter_Cabin_T'
    ],
    'Coefficient': [
        -0.060693, -1.051405, -0.027040, -0.419895, -0.293378, 0.053657,
        0.236501, 1.345416, 1.074084, -1.458045, 0.940368, -2.031174, 0.848491,
        0.881867, 0.826806, 0.176881, 1.102238, 0.519373, 0.158335, -0.835780, -0.387565
    ]
}

df = pd.DataFrame(data)
df['Odds Ratio'] = np.exp(df['Coefficient'])

# Streamlit UI
st.title("Logistic Regression: Coefficients & Odds Ratios")

st.subheader("Model Coefficients Table")
st.dataframe(df)

st.subheader("Make a Prediction")

# Create inputs dynamically
user_input = []
for feature in df['Feature']:
    value = st.number_input(f"{feature}", value=0.0)
    user_input.append(value)

# Prediction (Mock example — replace with your trained model)
if st.button("Predict"):
    # Dummy logistic regression setup
    model = LogisticRegression()
    # This assumes the model has been trained already (replace with joblib/pickle loading in real use)
    prediction = 1 / (1 + np.exp(-np.dot(df['Coefficient'], user_input)))
    st.success(f"Predicted Probability of Outcome: {prediction:.4f}")
