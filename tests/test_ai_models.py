from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os

# Adjust the path to access the config file from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from config import Config
from app.modules.clean_csv import read_csv_chunks

# Function to evaluate and print metrics for each model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
    
    # Print metrics
    print(f"\nModel: \033[92m{model.__class__.__name__}\033[0m\n")
    print("\033[94mAccuracy:\033[0m", accuracy)
    print("\033[94mMean Absolute Error (MAE):\033[0m", mae)
    print("\033[94mMean Absolute Percentage Error (MAPE):\033[0m", mape, "%\n")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model.__class__.__name__}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Function to load data, preprocess, and split
def load_and_preprocess_data():
    # Load the dataset (replace with the correct path to your CSV)
    df_chunks = read_csv_chunks(Config.CLEANED_CSV_FULL_PATH, Config.COLS_FOR_MODEL, Config.CHUNK_SIZE)
    df = pd.concat(df_chunks, ignore_index=True)
    
    # Handle NaN, duplicates, and outliers as needed
    df.dropna(inplace=True)  # Example of handling NaNs (customize as necessary)
    
    # Encode 'pnns_groups_1' using LabelEncoder
    label_encoder_pnns = LabelEncoder()
    df['pnns_groups_1'] = label_encoder_pnns.fit_transform(df['pnns_groups_1'])
    
    # Ordinal encoding for the target variable 'nutriscore_grade'
    ordinal_encoder_grade = OrdinalEncoder(categories=[['e', 'd', 'c', 'b', 'a']])
    df['nutriscore_grade'] = ordinal_encoder_grade.fit_transform(df[['nutriscore_grade']])
    
    return df, label_encoder_pnns, ordinal_encoder_grade

# Load data, preprocess, and split
def load_and_prepare_data():
    df, label_encoder_pnns, ordinal_encoder_grade = load_and_preprocess_data()
    X = df.drop(columns="nutriscore_grade")
    y = df['nutriscore_grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_encoder_pnns, ordinal_encoder_grade

# List of models to test
models = [
    RandomForestClassifier(n_estimators=100, random_state=42, verbose=1),
    LogisticRegression(max_iter=2000, random_state=42, verbose=1),
    SVC(kernel="linear", max_iter=20000, random_state=42, verbose=True),
    KNeighborsClassifier(n_neighbors=5),  # KNeighborsClassifier does not support verbose output
    GradientBoostingClassifier(random_state=42, verbose=1)
]

# Run models and evaluate
X_train, X_test, y_train, y_test, label_encoder_pnns, ordinal_encoder_grade = load_and_prepare_data()

for model in models:
    print(f"\n\033[93mTraining model:\033[0m {model.__class__.__name__}\n")
    evaluate_model(model, X_train, X_test, y_train, y_test)

# Save the chosen model
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)
joblib.dump(final_model, 'app/ai-model/model.pkl')
joblib.dump(label_encoder_pnns, 'app/ai-model/label_encoder_pnns.pkl')
joblib.dump(ordinal_encoder_grade, 'app/ai-model/ordinal_encoder_grade.pkl')
