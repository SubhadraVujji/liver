'''import streamlit as st
import pickle
import pandas as pd

# Features list
features = ['Age', 'ALT', 'BMI', 'DM.IFG', 'FBG', 'GGT', 'TG', 'AST.PLT']
zero_allowed = ['DM.IFG']  # Only this feature can be 0

# Load model and scaler
with open("fibrosis.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# App title
st.title("A Machine Learning based Combined Approach for Liver Fibrosis Diagnosis in NAFLD Using Biomarkers and Demographics")
st.markdown("Enter the patient test values below to predict fibrosis stage:")

# Input fields
user_inputs = {}
for feature in features:
    user_inputs[feature] = st.text_input(f"{feature}:")

# Predict
if st.button("Predict"):
    # Check all fields are filled
    if all(user_inputs[feature].strip() != '' for feature in features):
        try:
            input_values = [float(user_inputs[feature]) for feature in features]
            input_dict = dict(zip(features, input_values))

            # Check for 0s in non-zero-allowed fields
            invalid_features = [f for f in features if f not in zero_allowed and input_dict[f] == 0]

            if invalid_features:
                st.warning(f"The following fields cannot be 0: {', '.join(invalid_features)}. Please enter valid values.")
            else:
                # Scale and Predict
                input_df = pd.DataFrame([input_values], columns=features)
                scaled_input = scaler.transform(input_df)
                prediction = model.predict(scaled_input)

                # Result
                fibrosis_stage = "Early Fibrosis" if str(prediction[0]).strip() == '0' else "Advanced Fibrosis"
                st.success(f"Predicted Fibrosis Stage: {fibrosis_stage}")

        except ValueError:
            st.error("All inputs must be numeric.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    else:
        st.warning("Please fill in all fields.")
'''

'''import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Define expected feature names
features = ['Age', 'ALT', 'BMI', 'DM.IFG', 'FBG', 'GGT', 'TG', 'AST.PLT']

# Load model, scaler, and threshold
with open("fibrosis.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("threshold.pkl", "rb") as f:
    threshold = pickle.load(f)

# Streamlit UI
st.title("A Machine Learning Based Approach for Liver Fibrosis Diagnosis in NAFLD")
st.markdown("### Enter patient details below:")

input_data = []
all_empty = True

# Collect user input
for feature in features:
    value = st.text_input(f"Enter {feature}:", value="")
    try:
        value = float(value) if value.strip() else 0.0
    except ValueError:
        st.error(f"Invalid input for {feature}. Please enter a numeric value.")
        st.stop()

    if value:
        all_empty = False
    input_data.append(value)

# Predict button
if st.button("Predict"):
    if all_empty:
        st.warning("Please enter values before making a prediction.")
    else:
        try:
            # Prepare input
            input_df = pd.DataFrame([input_data], columns=features)
            scaled_input = scaler.transform(input_df)

            # Predict probability
            prob = model.predict_proba(scaled_input)[0][1]  # Probability for Advanced
            prediction = "Advanced Fibrosis" if prob >= threshold else "Early Fibrosis"

            # Display prediction and confidence
            st.success(f"Predicted Fibrosis Stage: {prediction}")
            st.info(f"Model Confidence: {prob*100:.2f}% for Advanced Fibrosis")
            st.info(f"{(1 - prob)*100:.2f}% for Early Fibrosis")

            # Optional visual confidence bar
            st.progress(int(prob * 100))
        except Exception as e:
            st.error(f"Error: {e}")'''

import streamlit as st 
import pickle
import pandas as pd

# Features list
features = ['Age', 'ALT', 'BMI', 'DM.IFG', 'FBG', 'GGT', 'TG', 'AST.PLT']
zero_allowed = ['DM.IFG']  # Only this feature can be 0

# Load model and scaler
with open("fibrosis.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# App title
st.title("A Machine Learning based Combined Approach for Liver Fibrosis Diagnosis in NAFLD Using Biomarkers and Demographics")
st.markdown("Enter the patient test values below to predict fibrosis stage:")

# Input fields
user_inputs = {}
for feature in features:
    user_inputs[feature] = st.text_input(f"{feature}:")

# Predict
if st.button("Predict"):
    # Check all fields are filled
    if all(user_inputs[feature].strip() != '' for feature in features):
        try:
            input_values = [float(user_inputs[feature]) for feature in features]
            input_dict = dict(zip(features, input_values))

            # Check for 0s in non-zero-allowed fields
            invalid_features = [f for f in features if f not in zero_allowed and input_dict[f] == 0]

            if invalid_features:
                st.warning(f"The following fields cannot be 0: {', '.join(invalid_features)}. Please enter valid values.")
            else:
                # Scale input and predict
                input_df = pd.DataFrame([input_values], columns=features)
                scaled_input = scaler.transform(input_df)

                # Predict fibrosis stage
                prediction = model.predict(scaled_input)
                fibrosis_stage = "Early Fibrosis" if str(prediction[0]).strip() == '0' else "Advanced Fibrosis"
                st.success(f"Predicted Fibrosis Stage: {fibrosis_stage}")

                # Show confidence score
                probabilities = model.predict_proba(scaled_input)[0]
                confidence = max(probabilities) * 100
                st.info(f"Probability Score: {confidence:.2f}%")

        except ValueError:
            st.error("All inputs must be numeric.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    else:
        st.warning("Please fill in all fields.")
