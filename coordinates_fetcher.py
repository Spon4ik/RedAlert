import pandas as pd
from geopy.geocoders import GoogleV3
import time
import json
import os
from utils import save_json  # import function from utils.py

def read_api_key(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data['api_key']

def fetch_coordinates(df, api_key, existing_coords):
    geolocator = GoogleV3(api_key=api_key)
    
    # Get unique cities with missing lat/lng values
    missing_coords = df[df['City'].apply(lambda x: x not in existing_coords)].copy()

    coords = {}
    for city in missing_coords['City']:
        try:
            location = geolocator.geocode(city)
            if location:
                coords[city] = [location.latitude, location.longitude]
            else:
                coords[city] = [None, None]
            time.sleep(1)  # to avoid hitting the API rate limit
        except Exception as e:
            print(f"Error fetching coordinates for {city}: {e}")
            coords[city] = [None, None]
    
    return coords

def update_coordinates(df, api_key_path, coords_file_path):
    # Read the API key from the JSON file
    api_key = read_api_key(api_key_path)

    # Load the existing coordinates from the JSON file, if it exists
    if os.path.exists(coords_file_path):
        with open(coords_file_path, 'r', encoding='utf-8') as file:
            existing_coords = json.load(file)
    else:
        existing_coords = {}
    
    # Fetch coordinates for missing cities
    new_coords = fetch_coordinates(df, api_key, existing_coords)

    # Update the existing coordinates with the new ones
    existing_coords.update(new_coords)
    
    # Save the updated coordinates back to the JSON file
    save_json(existing_coords, coords_file_path)
    
    # Create a DataFrame from existing_coords
    coords_df = pd.DataFrame.from_dict(existing_coords, orient='index', columns=['lat', 'lng']).reset_index()
    coords_df.rename(columns={'index': 'City'}, inplace=True)
    
    # Update the original DataFrame with the new coordinates
    df.set_index('City', inplace=True)
    df.update(coords_df.set_index('City'))
    df.reset_index(inplace=True)

    return df