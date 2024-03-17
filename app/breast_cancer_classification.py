import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import pickle
from joblib import dump

# Load the dataset without 'id' column
data = pd.read_csv("app/data/breast-cancer.csv").drop(columns=['id'])

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Handling missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the NuSVC classifier model
nu_svm_classifier = NuSVC().fit(X_train, y_train)

# Make predictions and calculate accuracy
accuracy = accuracy_score(y_test, nu_svm_classifier.predict(X_test))
print("Model Accuracy:", accuracy)

# Save the model to a .pkl file
dump(nu_svm_classifier, 'breast_cancer.pkl')
