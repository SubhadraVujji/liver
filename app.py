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
st.title("A Machine Learning based Approach for Liver Fibrosis Diagnosis in NAFLD Using Biomarkers and Demographics")
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
                st.info(f"Probability/Score: {confidence:.2f}%")

        except ValueError:
            st.error("All inputs must be numeric.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    else:
        st.warning("Please fill in all fields.")
