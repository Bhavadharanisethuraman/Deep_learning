import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the trained model

model = tf.keras.models.load_model('deep_neural_network_model.keras')

# Load and preprocess the dat

# Encode categorical variables


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Streamlit App
st.title('Deep Neural Network Classifier for Thyroid Cancer Recurrence')

# Sidebar for user input features
st.sidebar.header('User Input Features')
def user_input_features():
    features = {}
    for i, col in enumerate(X.columns):
        features[col] = st.sidebar.number_input(col, value=0.0)
    return pd.DataFrame(features, index=[0])

input_data = user_input_features()

# Display input data
st.subheader('User Input Data')
st.write(input_data)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Make prediction
prediction = model.predict(input_data_scaled)
predicted_class = (prediction > 0.5).astype(int)

st.subheader('Prediction')
st.write('Predicted Probability:', prediction[0][0])
st.write('Predicted Class:', 'Positive' if predicted_class[0][0] == 1 else 'Negative')

# Evaluate the model on test data
'''st.subheader('Model Evaluation')
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# Confusion Matrix
st.write('Confusion Matrix:')
cm = confusion_matrix(y_test, y_pred_classes)
st.write(cm)

# Classification Report
st.write('Classification Report:')
report = classification_report(y_test, y_pred_classes, output_dict=True)
st.write(pd.DataFrame(report).transpose())





# Display accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
st.write(f'Accuracy: {accuracy}')'''
