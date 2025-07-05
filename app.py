import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

st.set_page_config(page_title="Wine Quality Classifier", layout="wide")

st.title("🍷 Wine Quality Classification App")
st.write("Upload a CSV file with wine chemical data and predict if it's a good quality wine.")

# 1. Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if 'quality' not in data.columns:
        st.error("The uploaded file must contain a 'quality' column.")
        st.stop()

    # Binary classification target
    data['quality_class'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)

    st.subheader("🔍 Data Preview")
    st.dataframe(data.head())

    # Show missing values
    st.subheader("🧼 Missing Values")
    st.write(data.isnull().sum())

    # Visualizations
    st.subheader("📊 Class Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='quality_class', data=data, ax=ax1)
    ax1.set_xlabel("Quality Class (0 = Not Good, 1 = Good)")
    ax1.set_title("Distribution of Wine Quality Classes")
    st.pyplot(fig1)

    st.subheader("📈 Correlation Matrix")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    # Train model
    st.subheader("⚙️ Model Training")
    if st.button("Train Random Forest Model"):
        X = data.drop(['quality', 'quality_class'], axis=1)
        y = data['quality_class']
        feature_names = X.columns

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"Model trained successfully! Accuracy: {acc:.2f}")

        # Classification report
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Feature importance
        importances = pipeline.named_steps['classifier'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        st.subheader("🌟 Feature Importance")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax3)
        st.pyplot(fig3)

        # Save model
        joblib.dump(pipeline, 'wine_quality_model.pkl')
        st.success("Model saved as 'wine_quality_model.pkl'")

# 2. Manual Prediction
if os.path.exists('wine_quality_model.pkl'):
    st.subheader("🧪 Try a Prediction")
    model = joblib.load('wine_quality_model.pkl')

    # Get feature names from the trained model
    try:
        features = model.named_steps['classifier'].feature_names_in_
    except:
        features = X.columns if 'X' in locals() else []

    input_data = []
    col1, col2 = st.columns(2)
    for i, feature in enumerate(features):
        col = col1 if i % 2 == 0 else col2
        val = col.number_input(f"{feature}", step=0.1)
        input_data.append(val)

    if st.button("Predict Wine Quality"):
        prediction = model.predict([input_data])[0]
        result = "🍷 Good Quality Wine" if prediction == 1 else "🚫 Not Good Quality"
        st.success(f"Prediction Result: {result}")
