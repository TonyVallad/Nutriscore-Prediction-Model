from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os

# Adjust the path to access the config file from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from config import Config
from app.modules.clean_csv import read_csv_chunks

# Custom transformer for label encoding
class LabelEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.label_encoder.fit(X)
        return self

    def transform(self, X, y=None):
        return self.label_encoder.transform(X).reshape(-1, 1)

# Load and preprocess data
def load_data():
    df_chunks = read_csv_chunks(Config.CLEANED_CSV_FULL_PATH, Config.COLS_FOR_MODEL, Config.CHUNK_SIZE)
    df = pd.concat(df_chunks, ignore_index=True)
    df.dropna(inplace=True)  # Handling NaNs
    return df

# Define pipelines
def create_pipeline(model):
    # Define the column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("label_encoder_pnns", LabelEncodingTransformer(), ["pnns_groups_1"]),
            ("scaler", StandardScaler(), df.columns.drop(["nutriscore_grade", "pnns_groups_1"]))
        ],
        remainder="passthrough"
    )
    
    # Construct the pipeline with preprocessor and model
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    return pipeline

# Function to evaluate and print metrics
def evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
    
    # Print metrics
    print(f"\nModel: \033[92m{pipeline.named_steps['classifier'].__class__.__name__}\033[0m\n")
    print("\033[94mAccuracy:\033[0m", accuracy)
    print("\033[94mMean Absolute Error (MAE):\033[0m", mae)
    print("\033[94mMean Absolute Percentage Error (MAPE):\033[0m", mape, "%\n")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {pipeline.named_steps['classifier'].__class__.__name__}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Prepare data and split
df = load_data()
X = df.drop(columns="nutriscore_grade")
y = OrdinalEncoder(categories=[['e', 'd', 'c', 'b', 'a']]).fit_transform(df[['nutriscore_grade']]).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of models to test in pipelines
models = [
    RandomForestClassifier(n_estimators=100, random_state=42, verbose=1, n_jobs=-1),
    LogisticRegression(max_iter=2000, random_state=42, verbose=1, n_jobs=-1),
    SVC(kernel="linear", max_iter=20000, random_state=42, verbose=True),
    KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    GradientBoostingClassifier(random_state=42, verbose=1),
    HistGradientBoostingClassifier(random_state=42, verbose=1)
]

# Run models and evaluate
for model in models:
    pipeline = create_pipeline(model)
    print(f"\n\033[93mTraining model:\033[0m {model.__class__.__name__}\n")
    evaluate_model(pipeline, X_train, X_test, y_train, y_test)

# Save the chosen model with pipeline
final_model_pipeline = create_pipeline(RandomForestClassifier(n_estimators=100, random_state=42))
final_model_pipeline.fit(X_train, y_train)
joblib.dump(final_model_pipeline, 'app/ai-model/model_pipeline.pkl')
