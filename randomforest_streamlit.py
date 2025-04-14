import streamlit as st
import pandas as pd
import pickle


# 1. Load the trained model (outside of the main function for efficiency)
try:
    with open('random_forest_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    st.write('Successfully loaded randomforest model through .pkl file')
except FileNotFoundError:
    st.error("Error: 'random_forest_model.pkl' not found.  Make sure you have trained and saved the model first.")
    st.stop()  # Stop the app if the model can't be loaded
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()  # Stop the app if there's an error

# Define the required columns (must match the training data's columns)
required_columns = ['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                   'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                   'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                   'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Age',
                   'Education_Basic', 'Education_Graduation', 'Education_Master',
                   'Education_PhD', 'Marital_Status_Alone', 'Marital_Status_Divorced',
                   'Marital_Status_Married', 'Marital_Status_Single',
                   'Marital_Status_Together', 'Marital_Status_Widow',
                   'Marital_Status_YOLO']


# 2. Create the Streamlit app
def main():
    st.title("Customer Segmentation Prediction through randomforest")

    # 3. Get user input using Streamlit widgets
    user_input = {}
    try:
        user_input['Income'] = st.number_input("Income", min_value=0.0, format="%f", value=0.0) # Use number_input for numeric values
        user_input['Kidhome'] = st.number_input("Kidhome", min_value=0, max_value=10, value=0, step=1, format="%d")  # Use step for integer values
        user_input['Teenhome'] = st.number_input("Teenhome", min_value=0, max_value=10, value=0, step=1, format="%d")
        user_input['Recency'] = st.number_input("Recency", min_value=0,  value=0, step=1, format="%d")
        user_input['MntWines'] = st.number_input("MntWines", min_value=0,  value=0, step=1, format="%d")
        user_input['MntFruits'] = st.number_input("MntFruits", min_value=0,  value=0, step=1, format="%d")
        user_input['MntMeatProducts'] = st.number_input("MntMeatProducts", min_value=0,  value=0, step=1, format="%d")
        user_input['MntFishProducts'] = st.number_input("MntFishProducts", min_value=0,  value=0, step=1, format="%d")
        user_input['MntSweetProducts'] = st.number_input("MntSweetProducts", min_value=0,  value=0, step=1, format="%d")
        user_input['MntGoldProds'] = st.number_input("MntGoldProds", min_value=0,  value=0, step=1, format="%d")
        user_input['NumDealsPurchases'] = st.number_input("NumDealsPurchases", min_value=0,  value=0, step=1, format="%d")
        user_input['NumWebPurchases'] = st.number_input("NumWebPurchases", min_value=0,  value=0, step=1, format="%d")
        user_input['NumCatalogPurchases'] = st.number_input("NumCatalogPurchases", min_value=0,  value=0, step=1, format="%d")
        user_input['NumStorePurchases'] = st.number_input("NumStorePurchases", min_value=0,  value=0, step=1, format="%d")
        user_input['NumWebVisitsMonth'] = st.number_input("NumWebVisitsMonth", min_value=0,  value=0, step=1, format="%d")
        user_input['Age'] = st.number_input("Age", min_value=0,  value=0, step=1, format="%d")
        user_input['Education_Basic'] = st.number_input("Education_Basic (0 or 1)", min_value=0, max_value=1, value=0, step=1, format="%d")
        user_input['Education_Graduation'] = st.number_input("Education_Graduation (0 or 1)", min_value=0, max_value=1, value=0, step=1, format="%d")
        user_input['Education_Master'] = st.number_input("Education_Master (0 or 1)", min_value=0, max_value=1, value=0, step=1, format="%d")
        user_input['Education_PhD'] = st.number_input("Education_PhD (0 or 1)", min_value=0, max_value=1, value=0, step=1, format="%d")
        user_input['Marital_Status_Alone'] = st.number_input("Marital_Status_Alone (0 or 1)", min_value=0, max_value=1, value=0, step=1, format="%d")
        user_input['Marital_Status_Divorced'] = st.number_input("Marital_Status_Divorced (0 or 1)", min_value=0, max_value=1, value=0, step=1, format="%d")
        user_input['Marital_Status_Married'] = st.number_input("Marital_Status_Married (0 or 1)", min_value=0, max_value=1, value=0, step=1, format="%d")
        user_input['Marital_Status_Single'] = st.number_input("Marital_Status_Single (0 or 1)", min_value=0, max_value=1, value=0, step=1, format="%d")
        user_input['Marital_Status_Together'] = st.number_input("Marital_Status_Together (0 or 1)", min_value=0, max_value=1, value=0, step=1, format="%d")
        user_input['Marital_Status_Widow'] = st.number_input("Marital_Status_Widow (0 or 1)", min_value=0, max_value=1, value=0, step=1, format="%d")
        user_input['Marital_Status_YOLO'] = st.number_input("Marital_Status_YOLO (0 or 1)", min_value=0, max_value=1, value=0, step=1, format="%d")
    except ValueError:
        st.error("Invalid input. Please enter numeric values only.")
        return  # Exit if there's a value error.

    # 4. Create a Pandas DataFrame from the user input
    try:
        user_df = pd.DataFrame([user_input])
        # Ensure all columns are present and in the correct order
        for col in required_columns:
            if col not in user_df.columns:
                user_df[col] = 0  # Fill with a default value.
        user_df = user_df[required_columns]  # Reorder columns to match training data
    except Exception as e:
        st.error(f"Error creating the DataFrame: {e}")
        return # Exit if there is an error.

    # 5. Make the prediction when the user clicks a button
    if st.button("Predict"):
        try:
            prediction = loaded_model.predict(user_df)
            st.write(f"Prediction: {prediction[0]}")
            # If you have probabilities, you can display them too:
            # if hasattr(loaded_model, "predict_proba"): #Check if the model supports probabilities
            #   proba = loaded_model.predict_proba(user_df)
            #   st.write(f"Probabilities: {proba}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


if __name__ == "__main__":
    main()