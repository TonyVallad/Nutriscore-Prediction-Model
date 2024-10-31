import os

import requests
from tqdm import tqdm

from config import Config


def fetch_countries_en():
    """
    Fetches a list of country names in English from a specified API URL.

    :return: List of country names in English if the response is JSON. If the response is not JSON or an error occurs, appropriate error messages are printed.
    """
    try:
        response = requests.get(Config.COUNTRIES_EN_API_URL)
        response.raise_for_status()  # Vérifie si la requête a réussi (status code 200)

        # Vérifie si la réponse est en JSON
        if response.headers['Content-Type'] == 'application/json':
            countries_en = response.json()
            countries_en_names = [country['name']['common'] for country in countries_en]
            return countries_en_names
        else:
            print("La réponse n'est pas en JSON")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Error occurred: {req_err}")
    except ValueError as json_err:
        print(f"Erreur de décodage JSON: {json_err}")


def clean_countries(dataframe, column_name):
    """
    Cleans country names in a DataFrame by splitting and exploding them, stripping extra spaces, and normalizing to lowercase.
    It then maps the cleaned country names to their English counterparts using an API.
    NaN values are replaced with a specified unknown string.
    The cleaned data is saved to a CSV file.

    :param dataframe: The DataFrame containing country names to be cleaned.
    :param column_name: The specific column in the DataFrame to be cleaned.
    :return: The cleaned DataFrame with country names processed to ensure consistency and validity.
    """
    # Divise et explode pour obtenir chaque pays comme une ligne séparée
    dataframe[column_name] = dataframe[column_name].str.split(',')
    dataframe = dataframe.explode(column_name)
    # Nettoie les espaces supplémentaires et normalise en minuscules
    dataframe[column_name] = dataframe[column_name].str.strip().str.lower()
    # Obtenez la liste des noms de pays en anglais
    countries_en_names = {country.lower(): country for country in fetch_countries_en()}
    # Applique le nettoyage avec une barre de progression
    tqdm.pandas(desc="Nettoyage des pays")
    dataframe[column_name] = dataframe[column_name].progress_apply(
        lambda x: countries_en_names.get(x, Config.UNKNOWN_STR)
    )
    # Gère les valeurs NaN en remplissant avec UNKNOWN_STR
    dataframe[column_name] = dataframe[column_name].fillna(Config.UNKNOWN_STR)

    # Enregistre les données nettoyées dans un fichier CSV
    output_path = os.path.join(Config.DIRECTORY_PATH, f"{column_name}.csv")
    dataframe[column_name].to_csv(output_path, index=False)

    return dataframe