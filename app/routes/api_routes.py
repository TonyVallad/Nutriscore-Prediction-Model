from flask import Blueprint, request, jsonify
import joblib
from config import Config
import pandas as pd

# Define the blueprint for the API
api_bp = Blueprint('api_bp', __name__)

# Load your model
model = joblib.load(Config.MODEL_PATH)
label_encoder_pnns = joblib.load('app/ai-model/label_encoder_pnns.pkl')
ordinal_encoder_grade = joblib.load('app/ai-model/ordinal_encoder_grade.pkl')
scaler = joblib.load('app/ai-model/scaler.pkl')

@api_bp.route('/api/v1/predict-nutriscore', methods=['POST'])
def predict_nutriscore():
    try:
        data = request.get_json()

        # Check for missing features and create DataFrame
        features = ["pnns_groups_1", "energy-kcal_100g", "fat_100g", "saturated-fat_100g", 
                    "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g", 
                    "sodium_100g", "fruits-vegetables-nuts-estimate-from-ingredients_100g"]

        for feature in features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        input_data = pd.DataFrame([data])

        # Encode 'pnns_groups_1' using the label encoder
        input_data["pnns_groups_1"] = label_encoder_pnns.transform(input_data["pnns_groups_1"])

        # Scale numerical features
        input_data_scaled = scaler.transform(input_data)

        # Predict Nutri-Score grade
        prediction = model.predict(input_data_scaled)[0]

        # Decode the prediction if necessary
        prediction_grade = ordinal_encoder_grade.inverse_transform([[prediction]])[0][0]

        return jsonify({"nutriscore_grade": prediction_grade})

    except Exception as e:
        return jsonify({"error": str(e)}), 500