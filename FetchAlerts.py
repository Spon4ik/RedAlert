# FetchAlerts.py

import requests
import json
import os
import datetime

lang = 'he'
json_file_path = f'./data/alarms_history_{lang}.json'
def fetch_data(from_date, to_date):
    # url = f"https://www.oref.org.il//Shared/Ajax/GetAlarmsHistory.aspx?lang={lang}&fromDate={from_date}&toDate={to_date}&mode=0"
    url = f"https://alerts-history.oref.org.il//Shared/Ajax/GetAlarmsHistory.aspx?lang={lang}&fromDate={from_date}&toDate={to_date}&mode=0"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return json.loads(response.content)
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def increment_date(date_obj, days):
    return date_obj + datetime.timedelta(days=days)

def decrement_date(date_obj, days):
    return date_obj - datetime.timedelta(days=days)

def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_last_date_from_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if data:
                last_date = datetime.datetime.strptime(data[-1]["date"], "%d.%m.%Y")
                return last_date
    return None
    
def get_existing_rids(file_path):
    ensure_directory_exists(file_path)  # Ensure directory exists
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump([], file, ensure_ascii=False, indent=4)
        return []

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return [record["rid"] for record in data]

def append_new_data_to_json(file_path, new_data, existing_rids):
    file_data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            file_data = json.load(file)
    
    new_entries = [record for record in new_data if record["rid"] not in existing_rids]
    file_data.extend(new_entries)

    # Sort the combined data by 'rid'
    sorted_data = sorted(file_data, key=lambda x: x['rid'])

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(sorted_data, file, ensure_ascii=False, indent=4)



existing_rids = get_existing_rids(json_file_path)  # Fetch existing rids from JSON file

start_date = get_last_date_from_json(json_file_path)
if not start_date:
    start_date = datetime.datetime.strptime("24.07.2014", "%d.%m.%Y")
else:
    # Ensure start_date is not None before decrementing
    start_date = decrement_date(start_date, 0)  # Start from the last date in file (including it)

end_date = datetime.datetime.now()
initial_days_range = 730  # Starting with a larger range
days_range = initial_days_range - 1
temp_days_range = days_range

current_date = start_date
while current_date < end_date:
    # temp_days_range = days_range
    records_fetched = False

    while not records_fetched:
        next_date = increment_date(current_date, temp_days_range)
        if next_date > end_date:
            next_date = end_date

        from_date_str = current_date.strftime("%d.%m.%Y")
        to_date_str = next_date.strftime("%d.%m.%Y")

        print(f"Fetching data from {from_date_str} to {to_date_str} (days_range: {temp_days_range + 1})")
        data = fetch_data(from_date_str, to_date_str)

        if data:
            record_count = len(data)
            print(f"Number of records: {record_count}")
            
            # Modify this part
            if record_count >= 1999:
                if temp_days_range > 0:
                    temp_days_range = max(0, temp_days_range // 100)
                else:
                    append_new_data_to_json(json_file_path, data, existing_rids)  # Pass existing rids here
                    records_fetched = True
                    # temp_days_range = max(1, temp_days_range // 2)
            else:
                append_new_data_to_json(json_file_path, data, existing_rids)  # Pass existing rids here
                records_fetched = True
                if record_count < 500:
                    temp_days_range = temp_days_range * 2
        else:
            print(f"No data found for {from_date_str} to {to_date_str}")
            temp_days_range = (temp_days_range+1) * 2  # Gradually increase days_range
            records_fetched = True

    current_date = increment_date(next_date, 1)
