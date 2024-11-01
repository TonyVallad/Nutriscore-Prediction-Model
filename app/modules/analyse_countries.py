import os

import requests
from tqdm import tqdm

from config import Config


def fetch_countries_en():
    """
    Fetches a list of country names in English from an external API.

    This function sends a GET request to the API specified by the COUNTRIES_EN_API_URL
    in the Config. It checks the response status and ensures that the response is in
    JSON format. If successful, it extracts the common name of each country from the
    response and returns them as a list.

    Returns:
        list: A list of country names in English if the request is successful and the
        response is in JSON format. Prints an error message otherwise.

    Exceptions:
        Prints error messages for HTTP errors, request exceptions, and JSON decoding errors.
    """

    # API handler
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
    Cleans the country names in a DataFrame.

    This function splits and explodes the country names in the specified column of the 
    DataFrame so that each country name becomes a separate row. It normalizes the country 
    names by trimming spaces and converting them to lowercase. It then matches these names 
    against a list of country names in English obtained from an external API, replacing 
    unmatched names with a default unknown string. The function handles NaN values by 
    filling them with the unknown string. Finally, it saves the cleaned column to a CSV 
    file.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing the country names.
        column_name (str): The name of the column to clean.

    Returns:
        pandas.DataFrame: The DataFrame with cleaned country names.
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