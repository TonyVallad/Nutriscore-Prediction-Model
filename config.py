import os

class Config:
    """
    Configuration class for the application, containing database and file paths,
    constant values, and settings.

    Attributes:
        SECRET_KEY: A string used for session management and encryption. Defaults
            to 'a_default_secret_key' if not set in environment variables.
        SQLALCHEMY_DATABASE_URI: The URI for the SQLAlchemy database. Defaults to
            'sqlite:///nutriscore.db' if not set in environment variables.
        SQLALCHEMY_TRACK_MODIFICATIONS: A boolean that specifies if SQLAlchemy should
            track modifications of objects. Defaults to False.
        DB_NAME: The name of the database file. Defaults to "nutriscore.db".
        DB_FULL_PATH: The full file path to the database, generated based on DB_NAME.
        TABLE_NAME: The name of the table where data will be stored. Defaults to
            "produits".
        ORIGINAL_CSV_NAME: The name of the original CSV file containing product data.
            Defaults to "en.openfoodfacts.org.products.csv".
        CLEANED_CSV_NAME: The name of the cleaned CSV file. Defaults to
            "openfoodfact_clean.csv".
        CSV_FULL_PATH: The full file path to the cleaned CSV file, generated based on
            CLEANED_CSV_NAME.
        CHUNK_SIZE: The size of the chunks in which the CSV file will be processed.
            Defaults to 10000.
        VIEW_NAME: The name of the database view. Defaults to 'products_view'.
        SELECTED_COLS: A list of column names to be selected from the original CSV
            file for processing.
        COLS_STAT: A list of column names used for statistical analysis.
        DIRECTORY_PATH: The relative path to the directory containing static files.
            Defaults to "../static/".
        FILE_NAME: The name of the original CSV file to be processed. Defaults to
            "en.openfoodfacts.org.products.csv".
        OUTPUT_NAME: The name of the output CSV file after cleaning. Defaults to
            'openfoodfact_clean.csv'.
        NUTRI_OK: A list of valid nutriscore grades. Defaults to ["a", "b", "c", "d",
            "e"].
        COUNTRIES_EN_COL: The name of the column containing country information in English.
            Defaults to "countries_en".
        COUNTRIES_EN_API_URL: The API URL to fetch country information. Defaults to
            "https://restcountries.com/v3.1/all".
        UNKNOWN_STR: A default string used to denote unknown values. Defaults to
            "Unknown".
    """
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_default_secret_key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///nutriscore.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DB_NAME = "nutriscore.db"

    APP_STATIC_PATH = "app/static/"
    DB_FULL_PATH = 'instance/' + DB_NAME
    TABLE_NAME = "produits"

    ORIGINAL_CSV_NAME = "en.openfoodfacts.org.products.csv"
    CLEANED_CSV_NAME = "openfoodfact_clean.csv"
    CLEANED_100G_CSV_NAME = "100g_clean.csv"
    ORIGINAL_CSV_FULL_PATH = APP_STATIC_PATH + ORIGINAL_CSV_NAME
    CLEANED_CSV_FULL_PATH = APP_STATIC_PATH + CLEANED_CSV_NAME

    CHUNK_SIZE = 10000
    VIEW_NAME = 'products_view'

    SELECTED_COLS = [
        "code",
        "product_name",
        "quantity",
        "brands",
        "categories",
        "ingredients_text",
        "nutriscore_score",
        "nutriscore_grade",
        "energy-kj_100g",
        "energy-kcal_100g",
        "fat_100g",
        "saturated-fat_100g",
        "omega-3-fat_100g",
        "omega-6-fat_100g",
        "sugars_100g",
        "added-sugars_100g",
        "fiber_100g",
        "proteins_100g",
        "salt_100g",
        "fruits-vegetables-nuts-estimate-from-ingredients_100g",
        "countries_en"
    ]
    COLS_STAT = [
        "nutriscore_score",
        "nutriscore_grade",
        "energy-kj_100g",
        "energy-kcal_100g",
        "fat_100g",
        "saturated-fat_100g",
        "omega-3-fat_100g",
        "omega-6-fat_100g",
        "sugars_100g",
        "added-sugars_100g",
        "fiber_100g",
        "proteins_100g",
        "salt_100g",
        "fruits-vegetables-nuts-estimate-from-ingredients_100g"
    ]
    COLS_100G = [
        "energy-kcal_100g",
        "fat_100g",
        "saturated-fat_100g",
        "omega-3-fat_100g",
        "omega-6-fat_100g",
        "sugars_100g",
        "added-sugars_100g",
        "fiber_100g",
        "proteins_100g",
        "salt_100g",
        "fruits-vegetables-nuts-estimate-from-ingredients_100g",
        "nutriscore_score",
    ]
    COLS_FOR_MODEL = [
        "pnns_groups_1",
        "energy-kcal_100g",
        "fat_100g",
        "saturated-fat_100g",
        "sugars_100g",
        "fiber_100g",
        "proteins_100g",
        "salt_100g",
        "sodium_100g",
        "fruits-vegetables-nuts-estimate-from-ingredients_100g",
        "nutriscore_grade"
    ]

    MODEL_PATH = "app/ai-model/model.pkl"
    ShowGraphs = False  # Set to True to show graphs during model training

    DIRECTORY_PATH = "../static/"
    OUTPUT_NAME = 'openfoodfact_clean.csv'

    NUTRI_OK = ["a", "b", "c", "d", "e"]
    COUNTRIES_EN_COL = "countries_en"
    COUNTRIES_EN_API_URL = "https://restcountries.com/v3.1/all"
    UNKNOWN_STR = "Unknown"