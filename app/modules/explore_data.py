import pandas as pd
from config import Config
from flask import current_app
from app.modules.clean_csv import read_csv_chunks
import logging

def load_dataframe():
    """
    Load a DataFrame from a large CSV file by reading it in chunks.

    This function reads a CSV file specified by the `Config.ORIGINAL_CSV_FULL_PATH` constant,
    using the `read_csv_chunks` function from the `app.modules.clean_csv` module. The CSV file
    is read in chunks of size specified by the `Config.CHUNK_SIZE` constant. Each chunk is stored
    in a list of DataFrames, which are then concatenated into a single DataFrame.

    Parameters:
    None

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the CSV file.
    """

    # Prints loading status in terminal
    print("\n\033[93mLoading dataframe...\033[0m")

    # Initialisation of dataframe
    products = None
    list_df = []

    try:
        # Import CSV into a DFs by chunks
        list_df = read_csv_chunks(Config.CLEANED_CSV_FULL_PATH, [], Config.CHUNK_SIZE)

        # Concatenate chunks in one DF
        products = pd.concat(list_df, ignore_index=True)

        # Save DF in app config within the app context
        with current_app.app_context():
            current_app.config['PRODUCTS_DF'] = products
            current_app.config['SEARCH_RESULTS_DF'] = products
            current_app.config['loading_dataframe_status']['complete'] = True
            logging.debug("Data loading completed and status set to True.")

    except Exception as e:
        # If an error occurs, ensure the status is updated to indicate failure
        with current_app.app_context():
            current_app.config['loading_dataframe_status']['complete'] = False
            logging.error(f"Error in load_dataframe: {e}")
    
    # Saves dataframe in config
    current_app.config['PRODUCTS_DF'] = products

    return products
