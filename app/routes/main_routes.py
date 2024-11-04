from flask import Blueprint, render_template, current_app, jsonify
from config import Config
from app.modules.forms import NutriScoreForm
from app.modules.explore_data import load_dataframe
from app.modules.create_ai_model import load_and_preprocess_data, train_model
import pandas as pd
import threading
import os
import joblib

main_bp = Blueprint('main', __name__)

# Check loading status
@main_bp.route('/loading-dataframe-status', methods=['GET', 'POST'])
def loading_dataframe_status_check():
    """
    Endpoint to check the loading status of the dataframe.

    This route handles both GET and POST requests to check the current status
    of the dataframe loading process. The status is stored in the application's
    configuration and indicates whether the loading has been completed.

    Returns:
        JSON: A JSON object containing the loading status.
    """

    return jsonify(current_app.config['loading_dataframe_status'])

# Home Route
@main_bp.route('/', methods=['GET', 'POST'])
def index():
    """
    The home route of the application, which renders the index.html template.

    Returns:
        HTML: The rendered index.html template.
    """

    # Send to index.html
    return render_template('index.html')

# Check if model exists
def model_exists():
    """
    Check if the trained machine learning model exists.

    Returns:
        bool: True if the model file exists, False otherwise.
    """

    model_path = os.path.join('app', 'ai-model', 'model.pkl')
    
    return os.path.exists(model_path)

# Predict Route
@main_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handles requests to predict the Nutri-Score for a product based on its nutritional information.

    This function checks if the required data and model components are loaded in the application
    configuration. If not, it loads them. It then initializes the prediction form and handles form
    submissions to predict the Nutri-Score of a product using a trained machine learning model.

    Returns:
        Renders the prediction form template with the form object and predicted score.
    """

    # Check if the dataframe is already loaded
    if 'PRODUCTS_DF' not in current_app.config:
        current_app.config['PRODUCTS_DF'] = load_dataframe()
    
    # Get the DataFrame from the app config
    products = current_app.config['PRODUCTS_DF']
    pnns_groups_list = sorted(products['pnns_groups_1'].dropna().unique())

    # Check if the model exists and load it using joblib
    model_path = os.path.join('app', 'ai-model', 'model.pkl')
    scaler_path = os.path.join('app', 'ai-model', 'scaler.pkl')
    label_encoder_path = os.path.join('app', 'ai-model', 'label_encoder_pnns.pkl')
    ordinal_encoder_path = os.path.join('app', 'ai-model', 'ordinal_encoder_grade.pkl')

    # If model or scaler does not exist, create and train the model
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        df, label_encoder_pnns, ordinal_encoder_grade = load_and_preprocess_data()
        train_model(df, label_encoder_pnns, ordinal_encoder_grade)

    # Load the stored model and components
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    ordinal_encoder = joblib.load(ordinal_encoder_path)

    # Initialize the form
    form = NutriScoreForm()
    form.pnns_groups_1.choices = [(group, group) for group in pnns_groups_list]

    # Initialize the predicted score
    predicted_score = None

    # Handle form submission
    if form.validate_on_submit():
        try:
            # Collect form data into a DataFrame
            input_data = {
                "pnns_groups_1": label_encoder.transform([form.pnns_groups_1.data])[0],
                "energy-kcal_100g": float(form.energy_kcal_100g.data),
                "fat_100g": float(form.fat_100g.data),
                "saturated-fat_100g": float(form.saturated_fat_100g.data),
                "sugars_100g": float(form.sugars_100g.data),
                "fiber_100g": float(form.fiber_100g.data),
                "proteins_100g": float(form.proteins_100g.data),
                "salt_100g": float(form.salt_100g.data),
                "sodium_100g": float(form.sodium_100g.data),
                "fruits-vegetables-nuts-estimate-from-ingredients_100g": float(form.fruits_vegetables_nuts_estimate_from_ingredients_100g.data),
            }

            # Ensure the input DataFrame columns align exactly with training
            input_df = pd.DataFrame([input_data])[Config.COLS_FOR_MODEL[:-1]]

            # Cast to appropriate data types to match training
            input_df = input_df.astype({
                "pnns_groups_1": 'int64',
                "energy-kcal_100g": 'float64',
                "fat_100g": 'float64',
                "saturated-fat_100g": 'float64',
                "sugars_100g": 'float64',
                "fiber_100g": 'float64',
                "proteins_100g": 'float64',
                "salt_100g": 'float64',
                "sodium_100g": 'float64',
                "fruits-vegetables-nuts-estimate-from-ingredients_100g": 'float64',
            })

            # Debugging - Print Column Details
            print("\n\033[94mColumn names in prediction DataFrame:\033[0m\n", input_df.columns.tolist(), "\n")
            print("\033[94mData types in prediction DataFrame:\033[0m\n", input_df.dtypes, "\n")

            # Normalize the input data
            input_scaled = scaler.transform(input_df)

            # Print normalized data for confirmation
            print("\033[94mNormalized Input Data:\033[0m")
            print(input_scaled, "\n")

            # Predict the Nutri-Score
            prediction = model.predict(input_scaled)
            nutriscore_grade = ordinal_encoder.inverse_transform(prediction.reshape(-1, 1))[0][0]
            predicted_score = nutriscore_grade

            # Print predicted score
            print("\033[94mPredicted Nutri-Grade:\033[0m", predicted_score, "\n")
        except Exception as e:
            # Print error in red
            print(f"\n\033[91mError during prediction:\033[0m {e}\n")
            predicted_score = "Error"

    return render_template('prediction_form.html', form=form, pnns_groups_list=pnns_groups_list, predicted_score=predicted_score)

# Loading Data Route
@main_bp.route('/loading_data')
def loading_data():
    """
    Set the status to indicate that loading has not yet completed, then start a new thread
    to run the load_dataframe function. This function is called when the user navigates to
    the /loading_data route.

    Returns:
        HTML: The loading_dataframe.html template.
    """

    # Set the status to indicate that loading has not yet completed
    current_app.config['loading_dataframe_status'] = {"complete": False}

    # Capture the current app context to pass it into the new thread
    app_context = current_app._get_current_object()

    # Function to run in a new thread
    def load_data_with_context():
        with app_context.app_context():
            load_dataframe()

    # Start the thread
    threading.Thread(target=load_data_with_context).start()

    return render_template('loading_dataframe.html')
