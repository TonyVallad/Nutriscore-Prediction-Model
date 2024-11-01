import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust the path to access the config file from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from config import Config
from app.modules.clean_csv import read_csv_chunks

def load_and_preprocess_data():
    """
    Loads the preprocessed data from the cleaned CSV file and encodes categorical variables using
    LabelEncoder and OrdinalEncoder.

    Returns:
        df (pandas.DataFrame): The preprocessed DataFrame
        label_encoder_pnns (sklearn.preprocessing.LabelEncoder): The LabelEncoder object for 'pnns_groups_1'
        ordinal_encoder_grade (sklearn.preprocessing.OrdinalEncoder): The OrdinalEncoder object for 'nutriscore_grade'
    """

    # Load the dataset using read_csv_chunks function and concatenate all chunks into a single DataFrame
    df_chunks = read_csv_chunks(Config.CLEANED_CSV_FULL_PATH, Config.COLS_FOR_MODEL, Config.CHUNK_SIZE)
    df = pd.concat(df_chunks, ignore_index=True)

    # Encode categorical variables
    label_encoder_pnns = LabelEncoder()
    df['pnns_groups_1'] = label_encoder_pnns.fit_transform(df['pnns_groups_1'])

    # Use OrdinalEncoder for the target variable 'nutriscore_grade'
    ordinal_encoder_grade = OrdinalEncoder(categories=[['e', 'd', 'c', 'b', 'a']])
    df['nutriscore_grade'] = ordinal_encoder_grade.fit_transform(df[['nutriscore_grade']])

    return df, label_encoder_pnns, ordinal_encoder_grade

def train_model(df, label_encoder_pnns, ordinal_encoder_grade):
    """
    Trains a machine learning model (RandomForestClassifier in this case) using the preprocessed data.

    Prints detailed information about the training DataFrame columns, splits the dataset into training and testing sets,
    normalizes the features, prints the training AI model, trains the model, evaluates the model, generates the confusion
    matrix, and plots the confusion matrix with labels.

    Saves the model, encoders, and scaler in 'app/ai-model'.
    """

    # Separate features and target variable
    X = df.drop(columns="nutriscore_grade")
    y = df['nutriscore_grade'].ravel()  # Convert target to 1D array

    # Print detailed information about the training DataFrame columns
    print("\n\033[94mTraining DataFrame Info:\033[0m")
    print(X.info(), "\n")
    print("\033[94mTraining DataFrame Head:\033[0m")
    print(X.head(), "\n")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Print training AI model
    print("\033[93mTraining AI model...\033[0m\n")

    # Train a model (RandomForestClassifier in this case)
    model = RandomForestClassifier(random_state=42, verbose=1)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print("\n\033[94mModel Accuracy:\033[0m\n", accuracy_score(y_test, y_pred), "\n")
    mae = mean_absolute_error(y_test, y_pred)
    print("\033[94mMAE:\033[0m\n", mae, "\n")
    print("\033[94mClassification Report:\033[0m\n", classification_report(y_test, y_pred))

    # Predict the training or test set labels
    y_train_pred = model.predict(X_train_scaled)

    # Generate the confusion matrix
    cm = confusion_matrix(y_train, y_train_pred)

    # Plot the confusion matrix with labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # Display the confusion matrix
    print("\n\033[94mConfusion Matrix:\033[0m\n", cm, "\n")

    # Save the model, encoders, and scaler in 'app/ai-model'
    save_model_and_encoders(model, scaler, label_encoder_pnns, ordinal_encoder_grade)

def save_model_and_encoders(model, scaler, label_encoder_pnns, ordinal_encoder_grade):
    """
    Save the trained model and encoders to the 'app/ai-model' directory.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        The trained model.
    scaler : sklearn.preprocessing.StandardScaler
        The scaler object.
    label_encoder_pnns : sklearn.preprocessing.LabelEncoder
        The label encoder for PNNS (Product Name in Native language).
    ordinal_encoder_grade : sklearn.preprocessing.OrdinalEncoder
        The ordinal encoder for the grade (A, B, C, D, E).

    Returns
    -------
    None
    """

    # Create the directory if it doesn't exist
    save_dir = os.path.join('app', 'ai-model')
    os.makedirs(save_dir, exist_ok=True)

    # Save the model and other components
    joblib.dump(model, os.path.join(save_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(label_encoder_pnns, os.path.join(save_dir, 'label_encoder_pnns.pkl'))
    joblib.dump(ordinal_encoder_grade, os.path.join(save_dir, 'ordinal_encoder_grade.pkl'))

    print(f"\033[93mModel and encoders saved to '{save_dir}'\033[0m\n")

# Handles direct execution of this script
if __name__ == "__main__":
    # Load and preprocess the data
    df, label_encoder_pnns, ordinal_encoder_grade = load_and_preprocess_data()

    # Train the model
    train_model(df, label_encoder_pnns, ordinal_encoder_grade)
