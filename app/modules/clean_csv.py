import pandas as pd
from tqdm import tqdm
from config import Config
from app.modules.analyse_countries import clean_countries

def read_csv_chunks(file_path, selected_columns, chunk_size):
    """
    Read a CSV file in chunks and return a list of DataFrame chunks.

    :param file_path: The path to the CSV file to be read.
    :param selected_columns: A list of column names to be read from the CSV file.
    :param chunk_size: The number of rows per chunk to be read from the CSV file.
    :return: A list of DataFrame chunks, each containing the selected columns from the CSV file.
    """

    # Printing reading csv chunks message
    print("\n\033[93mReading CSV chunks...\033[0m")

    # Initialisation de la liste pour stocker les morceaux sélectionnés
    selected_chunks = []

    # Lire le fichier CSV en chunks avec une barre de progression
    if selected_columns != []:
        chunk_iter = pd.read_csv(file_path,
                                sep="\t",
                                low_memory=False,
                                header=0,
                                chunksize=chunk_size,
                                on_bad_lines="skip",
                                usecols=selected_columns)
    else:
        chunk_iter = pd.read_csv(file_path,
                                sep="\t",
                                low_memory=False,
                                header=0,
                                chunksize=chunk_size,
                                on_bad_lines="skip")

    with tqdm(desc="Lecture du CSV " + file_path, unit='chunk') as pbar:
        for chunk in chunk_iter:
            selected_chunks.append(chunk)
            # Mise à jour de la barre de progression
            pbar.update(1)
            # (Optionnel) Afficher des informations supplémentaires dans la barre de progression
            pbar.set_postfix(rows=chunk.shape[0])

    return selected_chunks

def filter_and_clean_data(dataframes, selected_columns, cols_stat, nutri_ok):
    """
    Filter and clean a list of pandas DataFrame objects.

    :param dataframes: List of pandas DataFrame objects to be filtered and cleaned.
    :param selected_columns: List of column names to retain in the DataFrame after filtering.
    :param cols_stat: List of column names where missing values should be filled with 0.
    :param nutri_ok: List of acceptable 'nutriscore_grade' values to retain in the filtered DataFrame.
    :return: Concatenated and cleaned DataFrame comprising only the specified columns and filtered by acceptable 'nutriscore_grade' values.
    """

    print("\n\033[93mFiltering and cleaning data...\033[0m")

    # Filter rows where 'nutriscore_score' and 'nutriscore_grade' are not missing, and select specified columns
    list_df_not_na = [
        df[df[['nutriscore_score', 'nutriscore_grade']].notna().all(axis=1)][selected_columns]
        for df in dataframes if len(df) > 0  # Only process non-empty DataFrames
    ]

    # Concatenate all filtered DataFrames into a single DataFrame
    df_not_na = pd.concat(list_df_not_na, ignore_index=True)

    # Keep only rows with 'nutriscore_grade' values that are in the acceptable list (nutri_ok)
    df_not_na = df_not_na[df_not_na["nutriscore_grade"].isin(nutri_ok)]

    # Replace any missing values in columns listed in cols_stat with 0
    df_not_na[cols_stat] = df_not_na[cols_stat].fillna(0)

    # Clean the country names
    df_not_na = clean_countries(df_not_na, Config.COUNTRIES_EN_COL)

    return df_not_na

def read_and_clean_csv():
    """
    Clean a CSV file by reading it in chunks, filtering, and cleaning the data according to specified configurations,
    and then outputting the cleaned data to a new CSV file.

    :return: None
    """

    print("\n\033[93mReading and cleaning CSV...\033[0m")

    # Define the path to the original CSV file using configuration settings
    file_path = Config.DIRECTORY_PATH + Config.ORIGINAL_CSV_NAME

    # Read the CSV file in chunks using pre-defined configurations (columns to select and chunk size)
    chunks = read_csv_chunks(file_path,
                             Config.SELECTED_COLS,
                             Config.CHUNK_SIZE)

    # Filter and clean the data chunks based on selected columns, columns to fill with 0, and acceptable 'nutriscore_grade' values
    clean_data = filter_and_clean_data(chunks,
                                       Config.SELECTED_COLS,
                                       Config.COLS_STAT,
                                       Config.NUTRI_OK)

    # Define the path to save the cleaned CSV file using configuration settings
    cleaned_file_path = Config.DIRECTORY_PATH + Config.CLEANED_CSV_NAME

    # Save the cleaned DataFrame to a new CSV file with tab-separated values and without the index column
    clean_data.to_csv(cleaned_file_path, sep='\t', index=False)

    print("\n\033[93mCSV cleaned and read\033[0m")

if __name__ == '__main__':
   read_and_clean_csv()