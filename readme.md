### **<h1 align="center">Nutri-Score Prediction AI Model</h1>**

<p align="center">
  <img src="app/static/logo.png" alt="Nutri-Score Logo">
</p>

<p align="center">
  <a href="readme.md">English</a> | <a href="readme_fr.md">Français</a>
</p>

This project is a machine learning application designed to predict the Nutri-Score of food products based on various nutritional data. The application is developed in Python with a Flask-based interface, enabling local predictions and a dynamic web-based display of product data.

---

![Front-end Screenshot](app/static/screenshot-index.png)

---

### **Table of Contents**

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [API Usage](#api-usage)
5. [Model Details](#model-details)
6. [Technologies Used](#technologies-used)
7. [Project Structure](#project-structure)
8. [How the Project Works](#how-the-project-works)
9. [Future Improvements](#future-improvements)

---

### **Features**

- **Nutri-Score Prediction**: Predicts Nutri-Score grades using machine learning models trained on Open Food Facts data.
- **Data Exploration and Cleaning**: Uses various preprocessing techniques, such as handling missing values and outliers, to ensure model accuracy.
- **Dynamic Frontend with Pagination**: A web interface displaying product data, allowing users to search and filter by Nutri-Score and other nutritional characteristics.

---

### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/TonyVallad/Nutriscore-Prediction-Model.git
   cd Nutriscore-Prediction-Model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the data:
   - Go to [Open Food Facts](https://fr.openfoodfacts.org/data).
   - Download the CSV file with the product data and place it in the `static` folder.

4. Run the application:
   ```bash
   python run.py
   ```

---

### **Usage**

- Start the application and navigate to `http://127.0.0.1:5000/` in your browser.
- Use the interface to explore product data and predict Nutri-Score grades.

#### Key Routes

- **Homepage (`/`)**: Displays an introductory page with navigation options.
- **API Prediction Endpoint (`/api/v1/predict-nutriscore`)**: Allows users to make predictions via a separate API route.
- **Product Listing (`/training_data`)**: Browse and search products with pagination and filters.
- **Search Data**: Filter products based on categories and Nutri-Score grades.

![Front-end Screenshot](app/static/screenshot-data.png)

![Front-end Screenshot](app/static/screenshot-form.png)

---

### **API Usage**

The project includes an API endpoint for making Nutri-Score predictions. This allows for easy integration with other applications or external services.

- **Endpoint**: `/api/v1/predict-nutriscore`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Example JSON Payload:

```json
{
    "pnns_groups_1": "Sugary snacks",
    "energy-kcal_100g": 500,
    "fat_100g": 25,
    "saturated-fat_100g": 10,
    "sugars_100g": 40,
    "fiber_100g": 5,
    "proteins_100g": 6,
    "salt_100g": 0.5,
    "sodium_100g": 0.2,
    "fruits-vegetables-nuts-estimate-from-ingredients_100g": 15
}
```

#### Response

The API will return a JSON response containing the predicted Nutri-Score grade based on the input values.

Example:
```json
{
    "nutriscore_grade": "e"
}
```

Use this API to make Nutri-Score predictions programmatically by sending a POST request with the required nutritional data.

---

### Model Details

The application uses a `RandomForestClassifier` as the default model with the following configuration:

- **Random State**: `42`  
  Ensures reproducibility by controlling the randomness. Setting a fixed random state allows the model to produce the same results each time it’s trained.

- **Bootstrap**: `False`  
  When set to `False`, the model uses the entire dataset to build each tree rather than random sampling with replacement. This may lead to higher variance but can improve accuracy in some cases.

- **Max Depth**: `30`  
  Limits the maximum depth of each decision tree in the forest. A value of `30` restricts the tree growth, helping to prevent overfitting by reducing the model complexity.

- **Min Samples Leaf**: `1`  
  Sets the minimum number of samples required to be at a leaf node (a terminal node of the tree). With `1`, each leaf can represent a single sample, which can capture more detailed patterns in the data but might increase the risk of overfitting.

- **Min Samples Split**: `5`  
  Specifies the minimum number of samples required to split a node. A value of `5` means that nodes with fewer than 5 samples will not be split, helping control tree growth and reduce overfitting.

- **Number of Estimators**: `300`  
  Indicates the number of trees in the forest. With `300` estimators, the model is more robust, as it aggregates the predictions of more individual trees, improving generalization but also increasing computational requirements.

- **Verbose**: `1`  
  Controls the level of detail displayed in the console during training. A setting of `1` provides updates on the training progress, which can be useful for tracking longer training times with multiple trees.

- **Number of Jobs**: `-1`  
  Specifies the number of CPU cores used for training. Setting this to `-1` uses all available processors, speeding up the training process, especially for larger datasets.

You can experiment with additional models by updating `create_ai_model.py`. The trained model is saved in `app/ai-model/`, allowing for seamless integration and deployment within the application.

---

### **Technologies Used**

- **Python**: Core language for data processing and machine learning.
- **Flask**: Web framework for the application interface.
- **Scikit-learn**: Machine learning library for model development.
- **Jinja2**: Templating engine for dynamic HTML generation.

---

### **Project Structure**

```plaintext
Nutriscore-Prediction-Model/
│
├── app/
│   ├── ai-model/            # Model and related files
│   ├── routes/              # Core application routes (split by functionality)
│   ├── static/              # Static assets (CSV data, images, CSS)
│   ├── templates/           # HTML templates
│   ├── __init__.py          # Application factory
│
├── config.py                # Configuration file
├── requirements.txt         # Dependencies
├── run.py                   # Main entry point
└── README.md                # Project documentation
```

---

### **How the Project Works**

1. **Data Preparation**: Loads and preprocesses the Open Food Facts dataset, handling missing values, outliers, and scaling features.
2. **Model Training**: Trains a machine learning model to predict Nutri-Score based on selected features.
3. **Frontend Interface**: Displays a paginated list of products, allowing filtering and searching based on Nutri-Score and nutritional data.

---

### **Future Improvements**

- **Enhanced API**: Extend the API for more comprehensive Nutri-Score predictions and data management, including handling model information and status to monitor model performance and availability.
- **Statistics Template**: Add a dedicated template to display various graphs and metrics about the current model, as well as comparisons with other tested models.
- **Improved Data Preparation**: Enhance missing value handling, scaling, and feature engineering for cleaner, more robust input data.
- **Simplify Data Exploration**: Merge the `training_data` and `search_results` templates into a single template to streamline data exploration.
- **UI Enhancements**: Refine the prediction form and results display for a more user-friendly experience.

---

### **License**

This project is licensed under the MIT License. You are free to use, modify, and distribute this software in any project, personal or commercial, provided that you include a copy of the original license and copyright notice.

For more information, see the [LICENSE](LICENSE) file.