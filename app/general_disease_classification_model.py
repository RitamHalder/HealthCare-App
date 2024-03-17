import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.pipeline import Pipeline

# Function to train and evaluate the general disease classification model
def train_disease_classification_model():

    # Load data from CSV
    data = pd.read_csv("data/Heart Attack.csv")

    # Split data into features and target
    X = data.drop('class', axis=1)
    y = data['class']

    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Create pipeline for preprocessing and model training
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])

    # Define hyperparameters to tune
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Evaluate model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("General Disease Classification Model Accuracy:", accuracy)

    # Print classification report
    print(classification_report(y_test, y_pred))

    return best_model

# Call the function to train and evaluate the disease classification model
disease_model = train_disease_classification_model()
joblib.dump(disease_model, 'train_disease_classification_model.pkl')
