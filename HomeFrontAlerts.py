# HomeFrontAlerts.ipynb
# %%
from itables import init_notebook_mode
init_notebook_mode(all_interactive=True)
import time
import pandas as pd
import re
import json
import os
import csv
import requests
from utils import save_json, split_and_explode
from datetime import datetime, timedelta

# %%
from itables import show, init_notebook_mode
init_notebook_mode(all_interactive=True)
import itables.options as opt
opt.lengthMenu = [2, 5, 10, 20, 50,100,200,500]

# %%
lang = 'he'
json_file_path = f'./data/alarms_history_{lang}.json'

# %% [markdown]
# # fetch_data

# %%
%run FetchAlerts.py

# %% [markdown]
# # Load Data from Json

# %%
# Loading the data from JSON file
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Creating a DataFrame from the JSON data
df_alarm_ht_raw = pd.DataFrame(data)
# Convert 'date' to datetime
df_alarm_ht_raw['date'] = pd.to_datetime(df_alarm_ht_raw['date'], dayfirst=True)
df_alarm_ht_raw['alertDate'] = pd.to_datetime(df_alarm_ht_raw['alertDate'])

# Displaying the DataFrame
display(df_alarm_ht_raw)

# %% [markdown]
# # Split records to Separate areas

# %%
df_alarm_ht_split = split_and_explode(df_alarm_ht_raw,'data')

display(df_alarm_ht_split)

# %%
# Fixing city names manualy
# Dictionary of replacements
replacements = {
    'מרחב אשקלון':
        'אשקלון',
    'מרחב אשדוד':
        'אשדוד',
    # 'זןהר':'זוהר‭',
    # 'סיט': 'אשדוד סיטי',
    # 'אשדוד -יא': 'אשדוד',
    # 'Third City': 'Third New Detailed Name'
}

# Perform the replacement using regular expressions
for old, new in replacements.items():
    df_alarm_ht_split['data'] = df_alarm_ht_split['data'].str.replace(old, new, regex=True)

# Display the result for the specific rid
display(df_alarm_ht_split[df_alarm_ht_split['rid'] == 20])

# %% [markdown]
# # Loading defend area data

# %%
defend_areas_file_path = './data/defend_areas.json'

# Load the DataFrame from the JSON file
df_defend_areas = pd.read_json(defend_areas_file_path, orient='records', lines=True)
# Rename columns in df_defend_areas_from_json
df_defend_areas.columns = ['City', 'Defense Time', 'Polygon Number']

display(df_defend_areas)

# %%
import pandas as pd
import json

# Load cities.json
with open('data/cities.json', 'r', encoding='utf-8') as f:
    cities_data = json.load(f)

# Convert JSON to DataFrame
df_cities = pd.DataFrame(cities_data)

# Display the columns of df_cities to see what we have
print(df_cities.columns)

# Define the columns to merge from df_cities
predefined_columns = ['id', 'name', 'name_en',
                    #   'name_ru', 'name_ar', 'zone', 'zone_en', 'zone_ru', 'zone_ar', 'time', 'time_en', 'time_ru', 'time_ar', 'countdown', 
       'lat', 'lng', 'value'
    #    , 'shelters'
       ]  # Replace 'another_column' with actual column names you want

# Keep only the predefined columns
df_cities = df_cities[predefined_columns]

# Drop rows where id is 0
df_cities = df_cities[df_cities['id'] != 0]
display(df_cities[predefined_columns])

# %%
df_cities_split = split_and_explode(df_cities,'value')
display(df_cities_split)


# %% [markdown]
# # Joining alerts and defend area data

# %%
# Perform the join
df_alarm_ht_def_areas = pd.merge(df_alarm_ht_split, df_defend_areas, left_on='data', right_on='Polygon Number', how='left')

# Display the resulting DataFrame to check the join
# display(df_alarm_ht_def_areas.head())
display(df_alarm_ht_def_areas[df_alarm_ht_def_areas['rid'] == 20])

# %%
# Assuming df_joined is the DataFrame resulting from your previous join
# and it contains both 'City' from df_defend_areas_from_json and 'data' from df
# df_alarm_ht_def_areas['Coalesced_City_Data'] = df_alarm_ht_def_areas['City'].combine_first(df_alarm_ht_def_areas['data'])
df_alarm_ht_def_areas['City'] = df_alarm_ht_def_areas['City'].combine_first(df_alarm_ht_def_areas['data'])
# display(df_alarm_ht_def_areas)
display(df_alarm_ht_def_areas[df_alarm_ht_def_areas['rid'] == 20])

# %%
# Merge the DataFrames on 'value' from cities.json and 'Coalesced_City_Data' from df_joined_def_areas
df_alarm_ht_def_areas_city = df_alarm_ht_def_areas.merge(
    df_cities_split[predefined_columns], 
    how='left', 
    left_on='City', 
    right_on='value'
)

# Display the merged DataFrame
# display(df_alarm_ht_def_areas_city)
display(df_alarm_ht_def_areas_city[df_alarm_ht_def_areas_city['rid'] == 20])

# %%
print(df_alarm_ht_def_areas_city.columns)

# %% [markdown]
# # Filtering data for rockets only

# %%
# # Show rocket sirens only
# df_prep_02 = df_prep_01[df_prep_01['category'] == 1].copy()
df_alarm_ht_def_areas_city = df_alarm_ht_def_areas_city[df_alarm_ht_def_areas_city['category'] == 1]
display(df_alarm_ht_def_areas_city)

# %% [markdown]
# # Prepare unique city list for geoposition search

# %%
print(df_alarm_ht_def_areas_city.columns)


# %%
import pandas as pd
from coordinates_fetcher import update_coordinates
import shutil

# Define file paths
api_key_path = 'maps_api_geocode_key.json'
coords_file_path = 'data/city_to_coords.json'  # Correct path without '/mnt'

# Convert unique city names to a DataFrame
df_unique_cities = pd.DataFrame(df_alarm_ht_def_areas_city['City'].unique(), columns=['City'])

# Add 'lat' and 'lng' columns with NaN values
df_unique_cities['lat'] = pd.NA
df_unique_cities['lng'] = pd.NA

# Update the coordinates
df_unique_cities_enriched = update_coordinates(df_unique_cities, api_key_path, coords_file_path)

# Display the updated DataFrame
display(df_unique_cities_enriched)


# %%
# df_unique_cities_enriched.to_csv('./data/df_unique_cities_enriched.csv', index=False, encoding='utf-8-sig')

# %%
# Merge the coordinates into the original DataFrame
df_alarm_ht_city_latlng_ggl_api = df_alarm_ht_def_areas_city.merge(df_unique_cities_enriched, on='City', how='left', suffixes=('', '_new'))

# Update the original DataFrame's lat and lng with the new values
df_alarm_ht_city_latlng_ggl_api['lat'] = df_alarm_ht_city_latlng_ggl_api['lat_new'].combine_first(df_alarm_ht_city_latlng_ggl_api['lat'])
df_alarm_ht_city_latlng_ggl_api['lng'] = df_alarm_ht_city_latlng_ggl_api['lng_new'].combine_first(df_alarm_ht_city_latlng_ggl_api['lng'])

# Drop the temporary columns used for merging
df_alarm_ht_city_latlng_ggl_api.drop(columns=['lat_new', 'lng_new'], inplace=True)

# Assign the updated DataFrame back to the original variable
df_alarm_ht_def_areas_city = df_alarm_ht_city_latlng_ggl_api

# Display the updated DataFrame
display(df_alarm_ht_def_areas_city)

# %% [markdown]
# # Load existing coordinates if the file exists

# %%
# import geopandas as gpd

# # Load GeoJSON data into a GeoDataFrame
# geo_df = gpd.read_file(r'C:\\Users\\b_ser\\OneDrive\\Document\\GitHub\\RedAlert\\data\\GIS\\muni_vaadim.geojson')

# # Display the GeoDataFrame to check if it's loaded correctly
# print(geo_df.head())


# %%
# # List of columns to drop
# columns_to_drop = [
#     'Nafa1', 'Nafa2', 'Hearot', 'Eshkol_MPn', 'Sign_Date', 'Tikun1', 'Tikun2', 
#     'Tikun3', 'Tikun4', 'Tikun5', 'Tikun6', 'tikun7', 'tikun8', 'tikun9', 'tikun10', 
#     'tikun11', 'tikun12', 'tikun13', 'tikun14', 'tikun15', 'CR_PNIM', 'CR_LAMAS', 'CV_PNIM', 'CV_LAMAS', 'Machoz', 'Shape_Leng', 'Shape_Area',
#        'AreaSQM', 'Precision'
# ]

# # Drop the columns
# geo_df = geo_df.drop(columns=columns_to_drop, errors='ignore')

# # Check the remaining columns to ensure the drop was successful
# print(geo_df.columns)


# %%
# # Initial merge attempt on 'Vaad_Heb'
# merged_df = df_prep_02.merge(geo_df, left_on='Coalesced_City_Data', right_on='Vaad_Heb', how='left')

# # Create mask for entries that failed to join
# mask = merged_df['Muni_Heb'].isnull()

# # Use the mask on merged_df directly for the fallback merge
# fallback_merged_df = merged_df.loc[mask].merge(geo_df, left_on='Coalesced_City_Data', right_on='Muni_Heb', how='left')

# # Drop the initial unsuccessful join columns from fallback_merged_df before concatenation
# fallback_merged_df.drop(columns=['Muni_Heb_y', 'Vaad_Heb_y'], inplace=True)
# fallback_merged_df.rename(columns={'Muni_Heb_x': 'Muni_Heb', 'Vaad_Heb_x': 'Vaad_Heb'}, inplace=True)

# # Combine the original merged data with the fallback data
# final_merged_df = pd.concat([merged_df[~mask], fallback_merged_df])

# print(final_merged_df.columns)
# # Drop unwanted '_y' columns if they are redundant or not needed
# columns_to_drop = [col for col in final_merged_df.columns if col.endswith('_y')]
# final_merged_df.drop(columns=columns_to_drop, inplace=True)

# # Optionally, you can rename '_x' columns back to their original names if needed
# columns_to_rename = {col: col.rstrip('_x') for col in final_merged_df.columns if col.endswith('_x')}
# final_merged_df.rename(columns=columns_to_rename, inplace=True)

# # Print the updated DataFrame columns to confirm changes
# print(final_merged_df.columns)


# %%
# import pandas as pd

# # Print out the columns to visualize any duplicated names
# print("Columns before cleanup:", final_merged_df.columns.tolist())

# # Check for duplicate column names manually or automatically
# from collections import Counter
# column_counts = Counter(final_merged_df.columns)
# duplicated_columns = [col for col, count in column_counts.items() if count > 1]

# print("Duplicated columns:", duplicated_columns)

# # Assuming 'geometry_x' and 'geometry_y' are the resulting columns from the merge
# # We need to decide which geometry column to keep. This example keeps the first geometry column.
# # Drop the second occurrence of any duplicated column if they are indeed duplicates. Adjust as necessary.
# for col in duplicated_columns:
#     # Show unique values if necessary to confirm duplicates
#     # print(final_merged_df[[col + '_x', col + '_y']].drop_duplicates())

#     # Drop the second occurrence. Rename first as needed.
#     final_merged_df.drop(columns=[col], inplace=True)  # Adjust to select which to drop
#     # final_merged_df.rename(columns={col + '_x': col}, inplace=True)

# # Now rename columns if '_x' or '_y' suffixes were added incorrectly
# columns_to_rename = {col: col.rstrip('_x') for col in final_merged_df.columns if col.endswith('_x')}
# final_merged_df.rename(columns=columns_to_rename, inplace=True)

# # Check columns after cleanup
# print("Columns after cleanup:", final_merged_df.columns.tolist())


# %%
# # Initial merge attempt on 'Vaad_Heb'
# merged_df = df_prep_02.merge(
#     geo_df, 
#     left_on='Coalesced_City_Data', 
#     right_on='Vaad_Heb', 
#     how='left', 
#     suffixes=('', '_geo')
# )

# # Create a mask for entries that failed to join based on 'Muni_Heb'
# mask = merged_df['Muni_Heb'].isnull()

# # Use the mask directly on merged_df for the fallback merge
# fallback_merged_df = merged_df[mask].merge(
#     geo_df,
#     left_on='Coalesced_City_Data', 
#     right_on='Muni_Heb', 
#     how='left', 
#     suffixes=('', '_fallback')
# )

# # Remove unnecessary '_geo' and '_fallback' columns if they exist
# columns_to_drop = [col for col in fallback_merged_df.columns if '_geo' in col or '_fallback' in col]
# fallback_merged_df.drop(columns=columns_to_drop, inplace=True)

# # Combine the original merged data with the fallback data
# final_merged_df = pd.concat([merged_df[~mask], fallback_merged_df])

# # Clean up column names if necessary
# # This step may need to be adjusted based on specific column handling needs
# columns_to_rename = {col: col.split('_')[0] for col in final_merged_df.columns if '_geo' in col}
# final_merged_df.rename(columns=columns_to_rename, inplace=True)

# # Print the final DataFrame columns to check for correctness
# print(final_merged_df.columns)


# %%
# display(final_merged_df)

# %%
# import geopandas as gpd
# from shapely.geometry import Polygon

# # Ensure that the GeoDataFrame is properly formed
# if not isinstance(final_merged_df, gpd.GeoDataFrame):
#     final_merged_df = gpd.GeoDataFrame(final_merged_df)

# # Explicitly reconstruct geometries to ensure they are 2D
# def convert_to_2d(geom):
#     if geom is None:
#         return None
#     if geom.geom_type == 'Polygon':
#         return Polygon([(x, y) for x, y, *rest in geom.exterior.coords])
#     return geom  # Handle other geometry types appropriately

# # Apply conversion to all geometries
# final_merged_df['geometry'] = final_merged_df['geometry'].apply(convert_to_2d)

# # Set the new geometry column as the active geometry column
# final_merged_df.set_geometry('geometry', inplace=True)

# # Check and set CRS if not set, assuming WGS 84
# if final_merged_df.crs is None:
#     final_merged_df.set_crs(epsg=4326, inplace=True)

# # Drop rows with None geometries, if any remain
# final_merged_df.dropna(subset=['geometry'], inplace=True)

# # Display the first few entries to ensure they are correctly formatted
# print(final_merged_df['geometry'].head())


# %%
# import pandas as pd
# import geopandas as gpd
# from keplergl import KeplerGl

# # Load your GeoDataFrame
# geo_df = gpd.read_file(r'C:\Users\b_ser\OneDrive\Document\GitHub\RedAlert\data\GIS\muni_vaadim.geojson')

# # Assume df_prep_02 is already loaded and prepared

# # Create a mapping of 'Coalesced_City_Data' to 'Vaad_Heb' and 'Muni_Heb'
# key_map = {}
# for index, row in geo_df.iterrows():
#     if pd.notna(row['Vaad_Heb']):
#         key_map[row['Vaad_Heb']] = row['Vaad_Heb']
#     if pd.notna(row['Muni_Heb']):
#         key_map[row['Muni_Heb']] = row['Muni_Heb']

# # Apply the mapping to create a new 'merge_key' in df_prep_02
# df_prep_02['merge_key'] = df_prep_02['Coalesced_City_Data'].map(key_map)

# # Merge using this new 'merge_key'
# final_merged_df = df_prep_02.merge(geo_df, left_on='merge_key', right_on='Vaad_Heb', how='left')
# final_merged_df.update(
#     df_prep_02.merge(geo_df, left_on='merge_key', right_on='Muni_Heb', how='left')
# )

# # Ensure the result is a GeoDataFrame and handle geometries correctly
# final_merged_df = gpd.GeoDataFrame(final_merged_df, geometry='geometry')

# # Set CRS if it was lost during merging
# if final_merged_df.crs is None:
#     final_merged_df.set_crs(geo_df.crs, inplace=True)

# # Visualization with Kepler.gl
# map_2 = KeplerGl(height=600)
# map_2.add_data(data=final_merged_df, name='Merged Data')
# map_2


# %%
# from keplergl import KeplerGl
# # Create a new kepler.gl map
# map_1 = KeplerGl(height=800)

# # Add data to the map
# map_1.add_data(data=final_merged_df, name='Alert Data')



# %%
# import geopandas as gpd

# # Ensure final_merged_df is a GeoDataFrame
# if not isinstance(final_merged_df, gpd.GeoDataFrame):
#     final_merged_df = gpd.GeoDataFrame(final_merged_df)

# # Check for None geometries and remove those rows or handle them as needed
# final_merged_df = final_merged_df[final_merged_df['geometry'].notnull()]

# # Now convert the valid geometries to GeoJSON
# final_merged_df['geometry_json'] = final_merged_df['geometry'].apply(lambda x: x.__geo_interface__)


# %%
# from keplergl import KeplerGl

# # Create a Kepler.gl map
# map_1 = KeplerGl(height=800)

# # Add the data to the map, ensuring you use the cleaned or handled geometry data
# map_1.add_data(data=final_merged_df, name='Alert Data')


# %% [markdown]
# # Fitch polygon for city

# %%

# with open(coords_file_path, 'r') as file:
#     city_coords = json.load(file)

# enriched_data_path = './data/enriched_city_data.json'

# enriched_data = {}
# if os.path.exists(enriched_data_path):
#     try:
#         with open(enriched_data_path, 'r') as file:
#             enriched_data = json.load(file)
#     except json.JSONDecodeError:
#         enriched_data = {}

# for city, coords in city_coords.items():
#     if city not in enriched_data:
#         geojson_data = fetch_osm_polygon(coords[0], coords[1])
#         save_json({city: geojson_data}, enriched_data_path)


# %%
# import json

# # Load GeoJSON data
# with open('./data/enriched_city_data.json', 'r') as file:
#     geojson_data = json.load(file)


# %% [markdown]
# # Assinging coordinates to cities

# %%
# # Apply mapping on all data
# df_prep_02['outLat'] = df_prep_02['Coalesced_City_Data'].apply(lambda x: city_to_coords[x][0])
# df_prep_02['outLong'] = df_prep_02['Coalesced_City_Data'].apply(lambda x: city_to_coords[x][1])
# df_prep_02['alertDate'] = df_prep_02['alertDate'].astype(str)
# df_prep_02['date'] = df_prep_02['date'].dt.strftime('%Y-%m-%d')
# # display(df)

# %%
# # Apply mapping on all data
# df_prep_02.loc[:, 'outLat'] = df_prep_02['Coalesced_City_Data'].apply(lambda x: city_to_coords[x][0])
# df_prep_02.loc[:, 'outLong'] = df_prep_02['Coalesced_City_Data'].apply(lambda x: city_to_coords[x][1])
# df_prep_02.loc[:, 'alertDate'] = df_prep_02['alertDate'].astype(str)
# df_prep_02.loc[:, 'date'] = df_prep_02['date'].dt.strftime('%Y-%m-%d')
# display(df_prep_02)

# %%
# print(df_prep_02.dtypes)

# %%
# # Assuming df is your DataFrame after loading JSON data
# import numpy as np
# import pandas as pd

# # Function to calculate the average latitude and longitude
# def calculate_average_lat_long(group):
#     avg_lat = np.mean(group['outLat'])
#     avg_long = np.mean(group['outLong'])
#     return pd.Series({'avgLat': avg_lat, 'avgLong': avg_long})

# # Group by 'rid' and apply the function, excluding group columns from the operation
# agg_df = df_prep_02.groupby('rid', as_index=False, group_keys=False).apply(calculate_average_lat_long)

# # Merge this aggregated data back into the main DataFrame
# df_prep_03 = df_prep_02.merge(agg_df, on='rid').copy()

# display(df_prep_03)
# display(df_prep_03[df_prep_03['rid'] == 28859])

# %%
import numpy as np
import pandas as pd

# Function to calculate the average latitude and longitude, excluding the group key handling
def calculate_average_lat_long(group):
    avg_lat = np.mean(group['lat'])
    avg_long = np.mean(group['lng'])
    return pd.Series([avg_lat, avg_long])

# Group by 'rid' and apply the function
agg_df = df_alarm_ht_def_areas_city.groupby('rid').apply(calculate_average_lat_long)

# Rename columns correctly after apply
agg_df.columns = ['avgLat', 'avgLong']

# Reset index to turn 'rid' back into a column (since it becomes an index after groupby)
agg_df = agg_df.reset_index()

# Merge this aggregated data back into the main DataFrame
df_prep_03 = df_alarm_ht_def_areas_city.merge(agg_df, on='rid').copy()

display(df_prep_03)

display(df_prep_03[df_prep_03['rid'] == 28859])

# %%
print(df_prep_03.dtypes)

# %%
import pandas as pd
import numpy as np

# Function to calculate the distance from the group's average lat and long
def calculate_distance(row, avg_lat, avg_long):
    return np.sqrt((row['lat'] - avg_lat)**2 + (row['lng'] - avg_long)**2)

# Apply the function to each row in the DataFrame
df_prep_03['distance_from_avg'] = df_prep_03.apply(lambda row: calculate_distance(row, row['avgLat'], row['avgLong']), axis=1)

# Define a threshold for what you consider an outlier
# threshold = df_prep_03['distance_from_avg'].quantile(0.95) # for example, 95th percentile
threshold = 0.5

# Filter the DataFrame to only include outliers
outliers = df_prep_03[df_prep_03['distance_from_avg'] > threshold]
# display(outliers)
display(outliers)
# display(outliers[outliers['rid'] == 20])

# %%
from geopy.distance import geodesic
# Central points with coordinates
sources = {
    # 'North Gaza': (31.5600, 34.4953),
    'North Gaza': (31.540477181629036, 34.56594995210597),
    'South Gaza': (31.22513272230714, 34.267723050601994),
    'Central Gaza': ((31.540477181629036 + 31.22513272230714) / 2, (34.565949952105974953 + 34.2677230506019942951) / 2),
    'South Lebanon': (33.05939871583828, 35.35166595731204),
    'Yemen': (15.56361074880308, 42.8564640870823)
}

# Define thresholds for each source
thresholds = {
    'North Gaza': 110,
    'South Gaza': 110,
    'Central Gaza': 110,
    'South Lebanon': 110,
    'Yemen': 100000
}

# Function to calculate distance from given point
def calculate_distance(row, point):
    target_point = (row['avgLat'], row['avgLong'])
    return geodesic(point, target_point).km

# Calculate distances from each source
for source_name, source_coords in sources.items():
    df_prep_03[f'distance_from_{source_name.replace(" ", "_").lower()}'] = df_prep_03.apply(calculate_distance, point=source_coords, axis=1)

# %% [markdown]
# # df_prep_04

# %%
def determine_probable_source(row):
    # Calculate distances from each source and check against thresholds
    valid_sources = {source: distance for source, distance in 
                     {src: row[f'distance_from_{src.replace(" ", "_").lower()}'] for src in sources}.items() 
                     if distance <= thresholds[source]}
    
    # No valid sources within their thresholds
    if not valid_sources:
        return 'Unknown', 'Unknown', None, None

    # Find the closest valid source
    closest_source = min(valid_sources, key=valid_sources.get)
    closest_distance = valid_sources[closest_source]

    return closest_distance, closest_source, sources[closest_source][0], sources[closest_source][1]
df_prep_04 = df_prep_03.copy()
# Apply the function to determine the probable source
df_prep_04['closest_distance'], df_prep_04['source'], df_prep_04['fromlat'], df_prep_04['fromlng'] = zip(*df_prep_04.apply(determine_probable_source, axis=1))
display(df_prep_04)

# %%
# Check if there's any leftover datetime objects in the configuration
def convert_datetime(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
        # return obj.strftime('%Y-%m-%d')
    raise TypeError("Unserializable object {} of type {}".format(obj, type(obj)))

# %%
# print(df_prep_04.dtypes)

# %%
# Find records with unknown source in df_src_points
df_unknown_source = df_prep_04[df_prep_04['source'] == 'Unknown']

# Optionally, to view a few records with unknown sources
df_unknown_source.head()


# %%
def debug_determine_probable_source(row):
    print(f"Row Data: {row}")
    distances = {source: row[f'distance_from_{source.replace(" ", "_").lower()}'] for source in sources}
    print(f"Distances: {distances}")
    closest_source = min(distances, key=distances.get)
    print(f"Closest Source: {closest_source}")
    closest_distance = distances[closest_source]
    print(f"Closest Distance: {closest_distance}")

    if closest_distance <= thresholds[closest_source]:
        print("Source within threshold")
        return closest_distance, closest_source, sources[closest_source][0], sources[closest_source][1]
    else:
        print("Source beyond threshold")
        return 'Unknown', 'Unknown', None, None

# Check if unknown_sources_df is empty
if not df_unknown_source.empty:
    # Test with a sample row from Eilat
    sample_row = df_unknown_source.iloc[0]
    debug_determine_probable_source(sample_row)
else:
    print("There are no records with an unknown source.")


# %%
import numpy as np
import pandas as pd

# Assuming df_src_points is your DataFrame with geographic data
# Replace 'lat' and 'lng' with the actual column names if they are different
df_prep_04['lat'] = df_prep_04['lat'].apply(lambda x: np.nan if x > 90 or x < -90 else x)
df_prep_04['lng'] = df_prep_04['lng'].apply(lambda x: np.nan if x > 180 or x < -180 else x)

# Drop rows where latitude or longitude are NaN
df_prep_04.dropna(subset=['lat', 'lng'], inplace=True)
df_prep_04['lat'] = df_prep_04['lat'].astype(float)
df_prep_04['lng'] = df_prep_04['lng'].astype(float)
# Drop the columns with many NaN values
# df_prep_04.drop(columns=['City', 'Defense Time', 'Polygon Number'], inplace=True)
# Convert datetime fields in the DataFrame again to ensure all are string
for column in df_prep_04.select_dtypes(include=['datetime64']):
    df_prep_04[column] = df_prep_04[column].dt.strftime('%Y-%m-%d %H:%M:%S')
display(df_prep_04)

# %%
df_prep_04.to_csv('./data/df_prep_04.csv', index=False, encoding
                  ='utf-8-sig')

# %%
# save_json(df_prep_04, './data/df_prep_04.json')

# %%
# print(df_prep_04.dtypes)

# %%
# filtered_df = df_prep_04[df_prep_04['rid'] == 1718]
# display(filtered_df)

# %%
# Install instructions available at - https://docs.kepler.gl/docs/keplergl-jupyter#install
from keplergl import KeplerGl
import requests

# Define the filename for the config file
config_filename = 'kepler_config.json'

# Check if the config file exists
if os.path.exists(config_filename):
    # Load the config from the file
    with open(config_filename, 'r', encoding='utf-8') as file:
        config = json.load(file)
    print("Config loaded from file.")
else:
    # Use a default config or define it here
    # config = {'version': 'v1', 'config': {}}  # Replace with your default configuration
    print("Using default config.")

# Now, use 'config' variable for your KeplerGl map
# kepler_map = KeplerGl(config=config)

# %%
# Define the height and width for the map (in pixels)
map_height = 700  # for example, 600 pixels in height
map_width = 1400   # for example, 800 pixels in width

# Create the KeplerGl object with the specified size
kepler_map = KeplerGl(height=map_height, width=map_width, config=config)
kepler_map.add_data(data=df_prep_04, name='OrefSirens')
kepler_map


# %%


# # Now that these columns are dropped, add the cleaned data to the KeplerGl map
# kepler_map = KeplerGl(height=700, width=250, config=config)
# kepler_map.add_data(data=df_prep_04, name='OrefSirens')
# display(kepler_map)

# %%
# # Display statistics for floating-point columns to find anomalies like extreme values or NaNs
# print(df_prep_04.describe())

# # Check for NaNs specifically
# print(df_prep_04.isna().sum())


# %%
# Assuming kepler_map is your KeplerGl object
config = kepler_map.config

# Define the filename for the config file
config_filename = 'kepler_config.json'

# Save the config to a JSON file
with open(config_filename, 'w', encoding='utf-8') as file:
    json.dump(config, file, ensure_ascii=False, indent=4)

print(f"Config saved to {config_filename}")

# %%
# # Assuming kepler_map is your KeplerGl object
# html_output = kepler_map._repr_html_()
# with open('kepler_map.html', 'w', encoding='utf-8') as file:
#     file.write(html_output)

# # kepler_map.save_to_html(file_name='kepler_map.html')

# %%
# Assuming kepler_map is your KeplerGl object
html_output = kepler_map._repr_html_()

# Decode bytes to string if necessary
if isinstance(html_output, bytes):
    html_output = html_output.decode('utf-8')

# Save the HTML output to a file
with open('kepler_map.html', 'w', encoding='utf-8') as file:
    file.write(html_output)


# %%
exit()

# %%
import requests
from bs4 import BeautifulSoup

url = 'https://www.oref.org.il'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all script tags
scripts = soup.find_all('script')

for script in scripts:
    if script.string:
        if 'Ajax/Get' in script.string:
            print(script.string)


# %%
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
import re

def extract_js_endpoints(soup):
    endpoints = set()
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string:
            matches = re.findall(r'/Shared/Ajax/Get\w+\.aspx', script.string)
            for match in matches:
                endpoints.add(match)
    return endpoints

def extract_comment_endpoints(soup):
    endpoints = set()
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        matches = re.findall(r'/Shared/Ajax/Get\w+\.aspx', comment)
        for match in matches:
            endpoints.add(match)
    return endpoints

def check_endpoint(url):
    try:
        response = requests.get(url)
        if response.status_code == 200 and "404.html" not in response.url:
            return response.url
    except requests.exceptions.RequestException:
        return None
    return None

url = 'https://www.oref.org.il'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract potential endpoints from JavaScript and comments
js_endpoints = extract_js_endpoints(soup)
comment_endpoints = extract_comment_endpoints(soup)
all_endpoints = js_endpoints.union(comment_endpoints)

# Validate endpoints
base_url = "https://www.oref.org.il"
valid_endpoints = []

for endpoint in all_endpoints:
    full_url = f"{base_url}{endpoint}"
    valid_url = check_endpoint(full_url)
    if valid_url:
        valid_endpoints.append(valid_url)
        print(f"Valid endpoint: {valid_url}")

print("Valid endpoints found:")
for endpoint in valid_endpoints:
    print(endpoint)


# %%
import requests

endpoint = "https://www.oref.org.il/Shared/Ajax/GetAlertsHistoryAreaData.aspx"
base_params = {
    'lang': 'he',
}

# Common parameter names to test
param_names = [
    'id', 'startDate', 'endDate', 'area', 'type', 'region', 'city', 'district'
]

# Example values to test with the parameters
param_values = [
    1, '2022-01-01', '2022-12-31', 'North', 'Alert', 'South', 'Tel Aviv', 'District 1'
]

def test_params(endpoint, base_params, param_names, param_values):
    valid_params = []
    for name in param_names:
        for value in param_values:
            params = base_params.copy()
            params[name] = value
            try:
                response = requests.get(endpoint, params=params)
                if response.status_code == 200 and '~~~ ~~~ ~~~ ~~~ ~~~' not in response.text:
                    print(f"Valid parameters: {params}")
                    valid_params.append(params)
            except requests.exceptions.RequestException as e:
                print(f"Error with parameters {params}: {e}")
    return valid_params

valid_params = test_params(endpoint, base_params, param_names, param_values)

print("Valid parameters found:")
for params in valid_params:
    print(params)


# %%
import requests

endpoint = "https://www.oref.org.il/Shared/Ajax/GetAlertsHistoryAreaData.aspx"

# List of valid parameters
valid_params_list = [
    {'lang': 'he', 'id': 1},
    {'lang': 'he', 'id': '2022-01-01'},
    {'lang': 'he', 'id': '2022-12-31'},
    {'lang': 'he', 'id': 'North'},
    {'lang': 'he', 'id': 'Alert'},
    {'lang': 'he', 'id': 'South'},
    {'lang': 'he', 'id': 'Tel Aviv'},
    {'lang': 'he', 'id': 'District 1'},
    {'lang': 'he', 'area': 1},
    {'lang': 'he', 'area': '2022-01-01'},
    {'lang': 'he', 'area': '2022-12-31'},
    {'lang': 'he', 'area': 'North'},
    {'lang': 'he', 'area': 'Alert'},
    {'lang': 'he', 'area': 'South'},
    {'lang': 'he', 'area': 'Tel Aviv'},
    {'lang': 'he', 'area': 'District 1'},
    {'lang': 'he', 'type': 1},
    {'lang': 'he', 'type': '2022-01-01'},
    {'lang': 'he', 'type': '2022-12-31'},
    {'lang': 'he', 'type': 'North'},
    {'lang': 'he', 'type': 'Alert'},
    {'lang': 'he', 'type': 'South'},
    {'lang': 'he', 'type': 'Tel Aviv'},
    {'lang': 'he', 'type': 'District 1'},
    {'lang': 'he', 'city': 1},
    {'lang': 'he', 'city': '2022-01-01'},
    {'lang': 'he', 'city': '2022-12-31'},
    {'lang': 'he', 'city': 'North'},
    {'lang': 'he', 'city': 'Alert'},
    {'lang': 'he', 'city': 'South'},
    {'lang': 'he', 'city': 'Tel Aviv'},
    {'lang': 'he', 'city': 'District 1'}
]

# Function to check if the response contains useful data
def contains_useful_data(response_text):
    if '~~~ ~~~ ~~~ ~~~ ~~~' in response_text or not response_text.strip():
        return False
    return True

# Iterate through valid parameters and print responses with useful data
for params in valid_params_list:
    try:
        response = requests.get(endpoint, params=params)
        if response.status_code == 200 and contains_useful_data(response.text):
            print(f"Parameters {params} returned useful data:")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error with parameters {params}: {e}")


# %%
import requests
from datetime import datetime, timedelta

endpoint = "https://www.oref.org.il/Shared/Ajax/GetAlertsHistoryAreaData.aspx"

# List of valid parameters with more specific values
specific_params_list = [
    {'lang': 'he', 'id': '2023-01-01'},
    {'lang': 'he', 'id': '2023-06-15'},
    {'lang': 'he', 'id': '2024-01-01'},
    {'lang': 'he', 'area': 'North'},
    {'lang': 'he', 'area': 'South'},
    {'lang': 'he', 'area': 'Central'},
    {'lang': 'he', 'type': 'Alert'},
    {'lang': 'he', 'type': 'Warning'},
    {'lang': 'he', 'city': 'Jerusalem'},
    {'lang': 'he', 'city': 'Tel Aviv'},
    {'lang': 'he', 'district': 'District 1'}
]

# Function to check if the response contains useful data
def contains_useful_data(response_text):
    if '~~~ ~~~ ~~~ ~~~ ~~~' in response_text or not response_text.strip():
        return False
    return True

# Iterate through specific parameters and print responses with useful data
for params in specific_params_list:
    try:
        response = requests.get(endpoint, params=params)
        if response.status_code == 200 and contains_useful_data(response.text):
            print(f"Parameters {params} returned useful data:")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error with parameters {params}: {e}")

# Test a range of dates
start_date = datetime.strptime('2023-01-01', '%Y-%m-%d')
end_date = datetime.strptime('2023-12-31', '%Y-%m-%d')
current_date = start_date

while current_date <= end_date:
    params = {'lang': 'he', 'id': current_date.strftime('%Y-%m-%d')}
    try:
        response = requests.get(endpoint, params=params)
        if response.status_code == 200 and contains_useful_data(response.text):
            print(f"Parameters {params} returned useful data:")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error with parameters {params}: {e}")
    current_date += timedelta(days=1)


# %%
import requests
from datetime import datetime, timedelta

endpoint = "https://www.oref.org.il/Shared/Ajax/GetAlertsHistoryAreaData.aspx"

# List of valid parameters with more specific values and correct date format
specific_params_list = [
    {'lang': 'he', 'id': '01.01.2023'},
    {'lang': 'he', 'id': '15.06.2023'},
    {'lang': 'he', 'id': '01.01.2024'},
    {'lang': 'he', 'area': 'North'},
    {'lang': 'he', 'area': 'South'},
    {'lang': 'he', 'area': 'Central'},
    {'lang': 'he', 'type': 'Alert'},
    {'lang': 'he', 'type': 'Warning'},
    {'lang': 'he', 'city': 'Jerusalem'},
    {'lang': 'he', 'city': 'Tel Aviv'},
    {'lang': 'he', 'district': 'District 1'}
]

# Function to check if the response contains useful data
def contains_useful_data(response_text):
    # Adjust the condition based on the actual response structure
    if '~~~' in response_text or response_text.strip() == '':
        return False
    return True

# Iterate through specific parameters and print responses with useful data
for params in specific_params_list:
    try:
        response = requests.get(endpoint, params=params)
        if response.status_code == 200 and contains_useful_data(response.text):
            print(f"Parameters {params} returned useful data:")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error with parameters {params}: {e}")

# Test a range of dates with correct date format
start_date = datetime.strptime('01.01.2023', '%d.%m.%Y')
end_date = datetime.strptime('31.12.2023', '%d.%m.%Y')
current_date = start_date

while current_date <= end_date:
    params = {'lang': 'he', 'id': current_date.strftime('%d.%m.%Y')}
    try:
        response = requests.get(endpoint, params=params)
        if response.status_code == 200 and contains_useful_data(response.text):
            print(f"Parameters {params} returned useful data:")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error with parameters {params}: {e}")
    current_date += timedelta(days=1)



