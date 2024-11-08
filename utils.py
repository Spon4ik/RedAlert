# utils.py

import os
import json
import re
import pandas as pd
import shutil


# def save_json(data, file_path):
#     if os.path.exists(file_path):
#         with open(file_path, 'r', encoding='utf-8') as file:
#             try:
#                 existing_data = json.load(file)
#             except json.JSONDecodeError:
#                 existing_data = []
#     else:
#         existing_data = []
    
#     existing_data.append(data)
    
#     with open(file_path, 'w', encoding='utf-8') as file:
#         json.dump(existing_data, file, ensure_ascii=False, indent=4)

def save_json(data, file_path):
    temp_file_path = file_path + '.tmp'
    backup_file_path = file_path + '.bak'
    
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_file_path)  # Create a backup before modifying
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}
    
    existing_data.update(data)
    
    with open(temp_file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
    
    # Use os.replace to handle overwriting on Windows
    os.replace(temp_file_path, file_path)


import re
import pandas as pd

def split_and_explode(df, column_name):
    """
    Splits the specified column in the DataFrame based on specific rules and explodes the resulting lists into separate rows.
    Ensures that the output DataFrame does not contain duplicate records.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.
    column_name (str): The name of the column to be split and exploded.

    Returns:
    pd.DataFrame: A new DataFrame with the specified column split, exploded, and duplicates removed.
    """
    # Function to split 'data' based on specific rules
    def split_data(data):
        # Split only at commas followed by a space and a non-digit character (to avoid splitting sub-areas)
        return re.split(r', (?=\D)', data)

    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Apply the split function to the specified column
    df_copy[column_name] = df_copy[column_name].apply(split_data)

    # Explode the list into separate rows
    df_copy = df_copy.explode(column_name).reset_index(drop=True)

    # Strip whitespace from the exploded column
    df_copy[column_name] = df_copy[column_name].str.strip()

    # Remove duplicate records
    df_copy = df_copy.drop_duplicates()

    return df_copy


# import requests
# from bs4 import BeautifulSoup

# url = 'https://www.mivzaklive.co.il/התראת-צבע-אדום-מספרי-פוליגונים-וזמני-ה'

# response = requests.get(url)
# soup = BeautifulSoup(response.text, 'html.parser')

# # Find the div with the specified class and then find the table within this div
# container = soup.find('div', {'class': 'entry-content entry clearfix'})
# if container:
#     table = container.find('table')
#     if table:
#         rows = table.find_all('tr')
#         data = [[cell.text.strip() for cell in row.find_all('td')] for row in rows]
#         print(data)
#     else:
#         print("No table found within the specified div.")
# else:
#     print("No div with the specified class found.")
# import pandas as pd

# # Convert the list of lists (data) into a DataFrame
# df_defend_areas = pd.DataFrame(data[1:], columns=data[0])  # assuming the first sublist in data is the header

# # Display the DataFrame to verify its structure

# display(df_defend_areas)

