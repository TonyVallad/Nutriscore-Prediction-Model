import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Adjust the path to access the config file from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from config import Config
from app.modules.clean_csv import read_csv_chunks

def prepare_multiclass_outputs(model, X_test, y_test, classes):
    """
    Binarizes y_test and calculates predicted probabilities (y_score) for multi-class outputs.
    
    Parameters:
    - model: The trained classifier.
    - X_test: Scaled test features.
    - y_test: True labels for the test set.
    - classes: List of unique class labels for binarization.
    
    Returns:
    - y_test_binarized: Binarized test labels.
    - y_score: Predicted probabilities for each class.
    """

    # Ensure y_test is a valid array-like input for label_binarize
    y_test_binarized = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)  # Get predicted probabilities

    return y_test_binarized, y_score

def plot_confusion_matrix(y_train, y_train_pred):
    """
    Plots the confusion matrix for the training predictions.
    """

    print("\n\033[93mGenerating Confusion Matrix...\033[0m\n")

    cm = confusion_matrix(y_train, y_train_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
    
    print("\n\033[94mConfusion Matrix:\033[0m\n", cm, "\n")

def plot_roc_curve(y_test_binarized, y_score, n_classes, grade_labels):
    """
    Plots the ROC curve for each Nutri-Score grade based on binarized y_test and y_score.
    
    Parameters:
    - y_test_binarized: Binarized test labels.
    - y_score: Predicted probabilities.
    - n_classes: Number of classes (Nutri-Score grades).
    - grade_labels: Labels for each grade (from ordinal_encoder_grade).
    """

    print("\n\033[93mGenerating ROC Curve...\033[0m\n")

    # Compute ROC curve and ROC area for each grade
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'lime', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve (grade {grade_labels[i]}) (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Nutri-Score Grade')
    plt.legend(loc="lower right")
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plots the feature importance for the RandomForestClassifier model.
    """

    print("\n\033[93mGenerating Feature Importance...\033[0m\n")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.show()

def plot_precision_recall(y_test_binarized, y_score, n_classes, grade_labels):
    """
    Plots the Precision-Recall curve for each class.
    
    Parameters:
    - y_test_binarized: Binarized labels for the test set.
    - y_score: Predicted probabilities.
    - n_classes: Number of classes (Nutri-Score grades).
    - grade_labels: Labels for each grade (from ordinal_encoder_grade).
    """

    print("\n\033[93mGenerating Precision-Recall Curve...\033[0m\n")

    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test_binarized[:, i], y_score[:, i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'lime', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'Precision-Recall curve (grade {grade_labels[i]}) (area = {average_precision[i]:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Each Nutri-Score Grade")
    plt.legend(loc="lower left")
    plt.show()

def plot_learning_curve(model, X, y):
    """
    Plots the learning curve to evaluate if the model is overfitting or underfitting.
    """

    print("\n\033[93mGenerating Learning Curve...\033[0m\n")

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 8))
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Test score")
    plt.xlabel("Training Size")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.show()

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
    model = RandomForestClassifier(
        random_state=42,
        bootstrap = False,
        max_depth = 30,
        min_samples_leaf = 1,
        min_samples_split = 5,
        n_estimators = 300,
        verbose = 1,
        n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print("\n\033[94mModel Accuracy:\033[0m\n", accuracy_score(y_test, y_pred), "\n")
    mae = mean_absolute_error(y_test, y_pred)
    print("\033[94mMAE:\033[0m\n", mae, "\n")
    print("\033[94mClassification Report:\033[0m\n", classification_report(y_test, y_pred))

    # Predict the training or test set labels
    y_train_pred = model.predict(X_train_scaled)

    # Generate graphs if enabled in config
    if Config.ShowGraphs:
        # Plot Confusion Matrix
        plot_confusion_matrix(y_train, y_train_pred)

        # Prepare multiclass outputs for ROC and Precision-Recall curves
        y_test_binarized, y_score = prepare_multiclass_outputs(model, X_test_scaled, y_test, classes=[0, 1, 2, 3, 4])

        # Plot ROC Curve
        plot_roc_curve(y_test_binarized, y_score, len(ordinal_encoder_grade.categories_[0]), ordinal_encoder_grade.categories_[0])

        # Plot Feature Importance
        plot_feature_importance(model, X.columns)

        # Plot Precision-Recall Curve
        plot_precision_recall(y_test_binarized, y_score, len(ordinal_encoder_grade.categories_[0]), ordinal_encoder_grade.categories_[0])

        # Plot Learning Curve
        #plot_learning_curve(model, X_train_scaled, y_train)

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

    print(f"\n\033[92mModel and encoders saved to '{save_dir}'\033[0m\n")

# Handles direct execution of this script
if __name__ == "__main__":
    # Load and preprocess the data
    df, label_encoder_pnns, ordinal_encoder_grade = load_and_preprocess_data()

    # Train the model
    train_model(df, label_encoder_pnns, ordinal_encoder_grade)
