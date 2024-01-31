# streamlit_app.py

import streamlit as st
import joblib
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the trained SVM model
model = joblib.load('mental_health_recognition.pkl')


# Loading feature ranges dynamically
feature_ranges = {
    'anxiety_level': {'Low': '(0-10)', 'High': '(10-21)'},
    'depression': {'Low': '(0-13)', 'High': '(13-27)'},
    'sleep_quality': {'Low': '(0-2)', 'High': '(2-5)'},
    'stress_level': {'Low': '(0-1)', 'High': '(1-2)'},
    'social_support': {'Low': '(0-1)', 'High': '(1-3)'},
    'self_esteem': {'Low': '(0-15)', 'High': '(15-30)'},
    'future_career_concerns': {'Low': '(0-2)', 'High': '(2-5)'},
    'extracurricular_activities': {'Low': '(0-2)', 'High': '(2-5)'},
    'peer_pressure': {'Low': '(0-2)', 'High': '(2-5)'},
    'living_conditions': {'Low': '(0-2)', 'High': '(2-5)'},

}

# Loading the saved correlation matrix from the JSON file
with open('correlation_matrix.json', 'r') as json_file:
    corr_dict = json.load(json_file)

# Converting the dictionary back to a DataFrame
corr_matrix = pd.DataFrame.from_dict(corr_dict)

# Streamlit app
def main():
    st.title('Mental Health Prediction')

    # Introduction
    st.write(
        "This app predicts mental health status based on various features. "
        "Adjust the sliders to input your information and click 'Predict' to see the results."
    )
    st.subheader('How to Use:')
    st.write(
        "1. Adjust the sliders for each feature according to your information.\n"
        "2. Click on the 'Predict' button to see the predicted mental health status.\n"
        "3. Interpret the results and use the app as a tool for self-assessment."
    )

    # Creating input widgets for each feature
    st.markdown("## Feature Sliders")
    input_data = {}
    for feature, values in feature_ranges.items():
        low_limit, high_limit = map(lambda x: int(x.split('-')[1][:-1]), [values['Low'], values['High']])
        input_data[feature] = st.slider(
            f"{feature.replace('_', ' ').title()}:",
            min_value=int(values['Low'].split('-')[0][1:]),
            max_value=high_limit,
            step=1
    )

    input_df = pd.DataFrame([input_data])

    # Making predictions using the loaded SVM model
    if st.button('Predict'):
        prediction = model.predict(input_df)
        st.success(f'The predicted mental health status is: {int(prediction[0])}')
        input_df['mental_health_status'] = prediction[0]

        label = "Healthy" if prediction[0] == 0 else "Unhealthy"
        st.info(f'Interpretation: Predicted mental health status is {label}')

    st.markdown("---")

    # Displaying the feature ranges as a table
    st.subheader("Feature Ranges:")
    st.table(pd.DataFrame.from_dict(feature_ranges, orient='index', columns=['Low', 'High']))


    # Displaying the correlation matrix as a heatmap
    st.subheader("Correlation Matrix Heatmap")
    st.write("This heatmap illustrates the correlation between different features and their impact on the predicted mental health status. Values closer to 1 or -1 indicate a stronger correlation, while values closer to 0 suggest a weaker correlation."
             "A positive correlation suggests that as one feature increases, the other is likely to increase, and vice versa. Interpretation of the heatmap can provide insights into which features contribute more significantly to the mental health prediction.")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    st.pyplot(fig)

if __name__ == '__main__':
    main()

    st.markdown("---")
    st.markdown("<p style='font-size: 30px; font-family: Georgia, serif; font-weight: bold; color: #E91E63;'>"
                "Your mental health is a priority. Take a moment for self-reflection, practice self-care, "
        "and remember that it's okay not to be okay. You're stronger than you think."
        "</p>",
        unsafe_allow_html=True)

