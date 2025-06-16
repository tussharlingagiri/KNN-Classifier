"""
Healthcare k-NN Classifier
Author: Tusshar Lingagiri
Description: Implements a k-NN classifier to predict health outcomes using preprocessed healthcare data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
import os

# Define file paths
DATA_PATH = "data/Healthcare_Data_Preprocessed.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load the healthcare dataset."""
    try:
        data = pd.read_csv(DATA_PATH)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found. Ensure the dataset is in the 'data/' folder.")
        exit(1)

def preprocess_data(data):
    """Preprocess the dataset."""
    # Handle negative ages
    data["Age"] = data["Age"].apply(lambda x: 0 if x < 0 else x)

    # Numerical columns: Convert to numeric and fill missing values with median
    numerical_cols = [
        'Age', 'BMI', 'Blood_Pressure', 'Cholesterol', 'Glucose_Level', 
        'Heart_Rate', 'Sleep_Hours', 'Exercise_Hours', 'Water_Intake', 'Stress_Level'
    ]
    for col in numerical_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(data[col].median())

    # Categorical columns: Convert to integers and fill missing values with 0
    categorical_cols = [
        'Smoking', 'Alcohol', 'Diet', 'MentalHealth', 
        'PhysicalActivity', 'MedicalHistory', 'Allergies'
    ]
    for col in categorical_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)

    # Boolean columns: Map True/False to 1/0
    boolean_cols = ['Diet_Type_Vegan', 'Diet_Type_Vegetarian', 'Blood_Group_AB', 'Blood_Group_B', 'Blood_Group_O']
    for col in boolean_cols:
        data[col] = data[col].map({"True": 1, "False": 0}).fillna(0).astype(int)

    # Ensure target is integer
    data['Target'] = data['Target'].astype(int)

    return data

def train_and_evaluate(X, y):
    """Train and evaluate the k-NN model using cross-validation."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define k-fold cross-validation
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    # Evaluate k values from 1 to 50
    k_values = list(range(1, 51))
    mean_accuracies = []
    std_accuracies = []

    for k in k_values:
        pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        cv_results = cross_validate(pipeline, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1)
        mean_acc = cv_results['test_score'].mean()
        std_acc = cv_results['test_score'].std()
        mean_accuracies.append(mean_acc)
        std_accuracies.append(std_acc)
        print(f"k={k}: Mean Accuracy = {mean_acc:.4f} ± {std_acc:.4f}")

    # Find best k
    best_k = k_values[np.argmax(mean_accuracies)]
    best_accuracy = max(mean_accuracies)
    print(f"\nBest k: {best_k} with Mean Cross-Validation Accuracy: {best_accuracy:.4f}")

    # Train final model with best k
    final_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k))
    final_pipeline.fit(X_train, y_train)

    # Make sample predictions
    sample_predictions = pd.DataFrame({
        'Individual': [f'Person {i+1}' for i in range(5)],
        'Predicted_Health': final_pipeline.predict(X_test.head(5))
    })
    sample_predictions['Predicted_Health'] = sample_predictions['Predicted_Health'].map({0: 'Healthy', 1: 'Unhealthy'})

    # Save predictions to CSV
    sample_predictions.to_csv(os.path.join(OUTPUT_DIR, "sample_predictions.csv"), index=False)
    print("\nSample Predictions:")
    print(sample_predictions)

    return final_pipeline, best_k, best_accuracy

def main():
    """Main function to run the k-NN classifier."""
    print("Starting Healthcare k-NN Classifier...")
    
    # Load and preprocess data
    data = load_data()
    data = preprocess_data(data)

    # Split features and target
    X = data.drop('Target', axis=1)
    y = data['Target']

    # Train and evaluate model
    final_pipeline, best_k, best_accuracy = train_and_evaluate(X, y)

    print("\nModel training complete!")
    print(f"Results saved to '{OUTPUT_DIR}/sample_predictions.csv'")

if __name__ == "__main__":
    main()
