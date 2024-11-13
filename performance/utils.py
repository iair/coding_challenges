import os
import pandas as pd
import numpy as np
import re
import difflib
from typing import List, Any, Optional
import psycopg2

def standardize_column_names(columns: List[str]) -> List[str]:
    """
    Standardize column names by converting them to lowercase and removing special characters.

    Parameters
    ----------
    columns : List[str]
        The list of column names to standardize.

    Returns
    -------
    List[str]
        The list of standardized column names.
    """
    standardized_columns = pd.Series(columns).str.lower().str.replace('[^a-z0-9]', '', regex=True).tolist()
    return standardized_columns
def clean_text(text:str) -> str:
    """
    Clean the input text by converting it to lowercase and removing non-alphanumeric characters.

    Parameters
    ----------
    text (str) : The input text to be cleaned.

    Returns
    -------
    str: The cleaned text with only lowercase alphanumeric characters.
    
    Example:
    >>> clean_text("Progresivo 123!@#")
    'progresivo123'
    """

    return re.sub(r'[^a-z0-9]', '', text.lower())
def merge_similar_columns(dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Merge similar columns across all dataframes.

    This function takes a list of dataframes, finds columns with similar names, and renames them to the most similar name.
    The similarity is determined by the `difflib.SequenceMatcher` algorithm with a cutoff of 0.6.

    The function returns the list of dataframes with the columns renamed.

    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        The list of dataframes to merge similar columns

    Returns
    -------
    List[pd.DataFrame]
        The list of dataframes with similar columns merged
    """
    
    all_columns = set()
    for df in dataframes:
        all_columns.update(df.columns)
    
    column_mapping = {}
    for col in all_columns:
        matches = difflib.get_close_matches(col, all_columns, n=1, cutoff=0.6)
        if matches:
            column_mapping[col] = matches[0]
    
    for df in dataframes:
        df.rename(columns=column_mapping, inplace=True)
    
    return dataframes
def read_folder_data(folder_path: str) -> pd.DataFrame:
    """
    Reads all CSV and Excel files in the specified folder, standardizes their column names, 
    merges similar columns across all dataframes, and concatenates them into a single DataFrame.

    Parameters:
    folder_path : str
        The path to the folder containing the files to read.

    Returns:
    pd.DataFrame
        A single DataFrame with standardized column names and merged columns from all files.
    """
    dataframes = []
    for file in os.listdir(folder_path):
        if file.endswith(('.csv', '.xls', '.xlsx')):
            file_path = os.path.join(folder_path, file)
            if file.endswith('.csv'):
                df = pd.read_csv(file_path, header=0)
            else:
                df = pd.read_excel(file_path, header=0)
            df.columns = standardize_column_names(df.columns)
            dataframes.append(df)
    
    if dataframes:
        dataframes = merge_similar_columns(dataframes)
        return pd.concat(dataframes, ignore_index=True)
    else:
        return pd.DataFrame()
def null_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary DataFrame with the count and proportion of null values for each column.
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    Returns:
    pd.DataFrame: A DataFrame with columns 'null_count' and 'null_proportion' per each column.
    """
    total_rows = len(df)
    null_count = df.isnull().sum()
    null_proportion = null_count / total_rows
    summary_df = pd.DataFrame({
        'null_count': null_count,
        'null_proportion': null_proportion
    })
    
    return summary_df
def vectorized_numpy_approach(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a pandas DataFrame as input and returns a pandas DataFrame with a cleaned categoriaproducto column.
    It uses the numpy vectorize function to apply the clean_text function to all values in the categoriaproducto column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with a cleaned categoriaproducto column.
    """
    clean_text_vectorized = np.vectorize(clean_text)
    df['categoriaproducto'] = clean_text_vectorized(df['categoriaproducto'].values)
    return df
def numpy_grouping(df: pd.DataFrame, keywords=['progresivo', 'monofocal', 'bifocal', 'ocupacional', 'tratamiento']) -> pd.DataFrame:
    """
    Groups data based on unique values in 'nmerodealbarndecargoodealbarndeabono' and filters by keywords in 'categoriaproducto'.
    
    This function processes a DataFrame to produce a new DataFrame with four columns:
    - nmerodealbarndecargoodealbarndeabono: Unique values from the original column.
    - categoriaproducto: Grouped values of 'categoriaproducto' based on unique indices.
    - descripcindelproducto: Grouped values of 'descripcindelproducto' based on unique indices.
    - price: Sum of 'preciounitarionetosiniva' for rows where keywords are found in 'categoriaproducto'.

    Parameters:
    df (pd.DataFrame): The input DataFrame with necessary columns.
    keywords (list): Keywords to filter 'categoriaproducto'.

    Returns:
    pd.DataFrame: A DataFrame with the specified columns.
    """
    # Extract necessary columns as numpy arrays
    descripcindelproducto = df.descripcindelproducto.values
    categoriaproducto = df.categoriaproducto.values
    nmerodealbarndecargoodealbarndeabono = df.nmerodealbarndecargoodealbarndeabono.values
    prices = df.preciounitarionetosiniva.values

    # Create a boolean array where keywords are found in 'categoriaproducto'
    boolean_array = np.isin(categoriaproducto, keywords)

    # Find unique values and their inverse indices for grouping
    unique_nmeros, inverse_indices = np.unique(nmerodealbarndecargoodealbarndeabono, return_inverse=True)

    # Group 'categoriaproducto' values by unique inverse indices
    categoriaproducto_groups = [categoriaproducto[inverse_indices == i] for i in range(len(unique_nmeros))]

    # Group 'descripcindelproducto' values by unique inverse indices
    descripcindelproducto_groups = [descripcindelproducto[inverse_indices == i] for i in range(len(unique_nmeros))]

    # Calculate the sum of prices where keywords match in 'categoriaproducto'
    price_sums = [
        prices[(inverse_indices == i) & boolean_array].sum()
        for i in range(len(unique_nmeros))
    ]

    # Construct the resulting DataFrame
    return pd.DataFrame({
        'nmerodealbarndecargoodealbarndeabono': unique_nmeros,
        'categoriaproducto': categoriaproducto_groups,
        'descripcindelproducto': descripcindelproducto_groups,
        'price': price_sums
    })
def identify_keywords_and_positions(
    product_list: List[str],
    keywords: List[str]
) -> tuple[List[str], List[int]]:
    """
    Returns identified keywords and their positions in the list.

    Parameters:
    - product_list: list of product descriptions
    - keywords: list of keywords to look for

    Returns:
    - A tuple containing two lists: the first list contains the found keywords, and the second list contains their positions in the input list
    """
    keyword_positions = {
        keyword: [idx for idx, item in enumerate(product_list) if keyword in item.lower()]
        for keyword in keywords
        if any(keyword in item.lower() for item in product_list)
    }
    # Extract the found keywords and their positions
    found_keywords = list(keyword_positions.keys())
    found_positions = [pos for positions in keyword_positions.values() for pos in positions]
    return found_keywords, found_positions
def extract_descriptions(descriptions: List[Any], positions: List[int]) -> List[Any]:
    """
    Extracts values from the descriptions list at the specified positions.
    
    Parameters:
    - descriptions (List[Any]): List of descriptions in `descripcindelproducto`.
    - positions (List[int]): List of indices to extract values from.
    
    Returns:
    - List[Any]: List of extracted descriptions.
    """
    return [descriptions[pos] for pos in positions if pos < len(descriptions)]
def transform_description(description: str) -> str:
    """
    Transforms numbers in the description string according to specified rules:
    - Replaces numbers between 1.5 and 1.74 with "O" + number in hundreds (e.g., 1.5 -> O150).
    - Replaces numbers between 15 and 17 (with or without 'OR' prefix) with "O" + number in hundreds (e.g., OR15 -> O150).
    - Replaces numbers between 150 and 174 with "O" + the number (e.g., 150 -> O150).
    
    Parameters:
    - description (str): The input description string.
    
    Returns:
    - str: The transformed string, or None if no relevant numbers are found.
    """
    # Combined pattern for numbers with optional "OR" prefix
    match = re.search(r'\b(?:OR)?(1[,.][5-7][0-4]?|1[5-7]|1[5][0-9]|1[6][0-9]|17[0-4])\b', description, re.IGNORECASE)
    if match:
        num_str = match.group(1).replace(',', '.')  # Normalize any commas to periods
        num = float(num_str)
        
        # Format based on the range
        if 1.5 <= num <= 1.74:
            return f'O{int(num * 100):03d}'  # e.g., 1.5 -> O150, 1.6 -> O160
        elif 15 <= num <= 17:
            return f'O{int(num * 10):03d}'  # e.g., 15 -> O150, OR16 -> O160
        elif 150 <= num <= 174:
            return f'O{int(num):03d}'  # e.g., 150 -> O150, 167 -> O167
    return None
def extract_o_pattern(description: str) -> str:
    """
    Extracts the 'O' number pattern from the description string.
    
    Parameters:
    - description (str): The input description string.
    
    Returns:
    - str: The extracted 'O' pattern, or None if no pattern is found.
    """
    match = re.search(r'\bO(\d{3})\b', description, re.IGNORECASE)
    return f'O{match.group(1)}' if match else None

def filter_rows_by_keywords(
    df: pd.DataFrame, 
    keywords: List[str], 
    column_name: str = 'combined_keywords', 
    selected_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filters rows based on whether they contain all specified keywords in a specified column.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame to filter.
        keywords (List[str]): List of keywords to search for in each row.
        column_name (str, optional): The column name in which to search for keywords. Defaults to 'combined_keywords'.
        selected_columns (Optional[List[str]], optional): List of column names to return in the result. If None, all columns are returned.
    
    Returns:
        pd.DataFrame: A DataFrame containing only rows where all keywords are matched in the specified column.
    """
    # Create a boolean array that checks if all keywords are present in each row of the specified column
    boolean_array = np.array([all(word in lst for word in keywords) for lst in df[column_name].values])
    # Filter the DataFrame based on the boolean array
    if selected_columns:
        result = df.loc[boolean_array, selected_columns]
    else:
        result = df.loc[boolean_array]
    
    return result