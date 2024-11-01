import pandas as pd
from tqdm import tqdm
from config import Config
from app.modules.analyse_countries import clean_countries

def read_csv_chunks(file_path, selected_columns, chunk_size):
    """
    Reads a CSV file in chunks and optionally selects specific columns.

    This function reads a CSV file at the specified file path in chunks of a
    given size, using a progress bar to indicate the reading process. If a list
    of selected columns is provided, only those columns will be read from the
    file.

    Parameters:
    file_path (str): The path to the CSV file to be read.
    selected_columns (list): A list of columns to select from the CSV file. If
                             empty, all columns are read.
    chunk_size (int): The number of rows per chunk to be read from the CSV file.

    Returns:
    list: A list of DataFrames, each corresponding to a chunk of the CSV file.
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
    Filters and cleans the data by:

    1. Removing rows with missing 'nutriscore_score' and 'nutriscore_grade'.
    2. Selecting only the specified columns.
    3. Keeping only rows with 'nutriscore_grade' values that are in the acceptable list (nutri_ok).
    4. Replacing any missing values in columns listed in cols_stat with 0.
    5. Cleaning the country names.

    Parameters:
        dataframes (list): A list of DataFrames to be filtered and cleaned.
        selected_columns (list): List of column names to be selected.
        cols_stat (list): List of column names where missing values should be replaced with 0.
        nutri_ok (list): List of acceptable 'nutriscore_grade' values.

    Returns:
        pandas.DataFrame: The filtered and cleaned DataFrame.
    """

    # Prints filtering and cleaning data message
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
    Reads the original CSV file in chunks, filters and cleans the data by selecting only the specified columns,
    removing rows with missing 'nutriscore_score' and 'nutriscore_grade', keeping only rows with 'nutriscore_grade' values
    that are in the acceptable list (nutri_ok), replacing any missing values in columns listed in cols_stat with 0, and
    cleaning the country names. Finally, saves the cleaned DataFrame to a new CSV file with tab-separated values and
    without the index column.

    Parameters:
        None

    Returns:
        None
    """

    # Prints reading and cleaning csv message
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

# Handles direct execution of this script
if __name__ == '__main__':
   read_and_clean_csv()