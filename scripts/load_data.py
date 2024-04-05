"""
load and process the csv data files downloaded from USGS.gov
needs to be commented and explained
"""

import pandas as pd
import multiprocessing
import os

from functools import partial


def load_files(file_str,folder_path="../ressources/usgs"):
    all_files = os.listdir(folder_path)
    matching_files = [file for file in all_files if (file_str in file.lower() and file.lower().endswith(".csv"))]
    return sorted(matching_files)

def filter_files(start_year, end_year,files_list):
    files_list_without_csv = [filename.rstrip('.csv') for filename in files_list]
    matching_files = []
    for i in range(len(files_list_without_csv)):
        year = int(files_list_without_csv[i].rsplit('_')[1])
        if start_year <= year <= end_year:
            matching_files.append(files_list[i])
    return matching_files


def load_csv_file(file_path, raw_data):
    if raw_data:
        return pd.read_csv(file_path)
    else:
        return pd.read_csv(file_path, usecols=['id', 'time', 'type', 'mag', 'place', 'latitude', 'longitude', 'depth', "horizontalError", "depthError", "magError"])

def process_csv_file(raw_data, file_path):
    df = load_csv_file(file_path, raw_data)

    if not raw_data:
        df = df.drop(df.columns[df.columns.str.contains('Unnamed', case=False)], axis=1)

    df = df[df["type"] == "earthquake"]
    df = df.dropna(subset=['mag'])

    if not raw_data:
        df = df.drop_duplicates(subset="id", keep="last")
        df = df.drop_duplicates(subset="time", keep="last")
    return df

def area_custom_name(row):
    if isinstance(row, str):
        if 'Sumatra' in row:
            return 'Sumatra'
        if 'Alaska' in row:
            return 'Alaska'
        if 'region' in row:
            row = row.replace('region', '')
        if 'earthquake sequence' in row:
            row = row.replace('earthquake sequence ', '')
        if 'earthquake' in row:
            row = row.replace('earthquake ', '')
        if 'sequence' in row:
            row = row.replace('sequence ', '')
        if 'central' in row:
            row = row.replace('central ', '')
        if 'Northern' in row:
            row = row.replace('Northern ', '')
        if 'Northwestern' in row:
            row = row.replace('Northwestern ', '')
        if 'southeast' in row:
            row = row.replace('southeast ', '')
        if 'southern' in row:
            row = row.replace('southern ', '')
        if 'southwestern' in row:
            row = row.replace('southwestern', '')
        if 'western' in row:
            row = row.replace('western ', '')

        if 'of the ' in row:
            row = row.rsplit('of the')[1].strip()
        if 'of ' in row:
            row = row.rsplit('of ')[1].strip()
        if 'the ' in row:
            row = row.rsplit('the ')[1].strip()
        row = row.strip()
    return row

def process_file_wrapper(args):
    return process_csv_file(args)

def load_earthquakes_data(start_year, end_year, files_list, folder_path="../ressources/usgs", raw_data = False):
    # if we want to display raw data, we should set raw_data as True.
    # By default, raw_data is False.
    eq_files = filter_files(start_year, end_year, files_list)
    print(f"using {len(eq_files)} files over {end_year-start_year+1} years.\n")

    earthquake_db = pd.DataFrame()

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # Use all available CPU cores
    # Use functools.partial to create a wrapper function with a single argument
    process_file_wrapper = partial(process_csv_file, raw_data)
    result = pool.map(process_file_wrapper, [os.path.join(folder_path, file) for file in eq_files])
    earthquake_db = pd.concat(result)

    earthquake_db["date"] = pd.to_datetime(earthquake_db['time']).dt.date
    earthquake_db["timestamp"] = pd.to_datetime(earthquake_db['time']).apply(lambda x: x.timestamp())

    earthquake_db["nearest_area"] = earthquake_db['place'].str.split(',').str[-1]
    earthquake_db['nearest_area'] = earthquake_db['nearest_area'].apply(area_custom_name)

    if not raw_data:
        earthquake_db = earthquake_db.drop(["type","place"], axis=1)
        columns_to_convert = ["latitude", "longitude", "depth", "mag", "horizontalError", "depthError", "magError"]
        earthquake_db[columns_to_convert] = earthquake_db[columns_to_convert].apply(pd.to_numeric, errors='coerce') #coerce to transform non numeric value to NaN

    return earthquake_db

def load_countries_continent(folder_path="../ressources/population"):

    if not os.path.exists(folder_path): #should already exist as we have file
        os.makedirs(folder_path)

    dataframe = pd.read_csv(os.path.join(folder_path,'Countries-Continents.csv'), sep = ',', na_filter=False )
    add_sumatra = pd.DataFrame({'Continent': ['Asia'], 'Country': ['Sumatra']})
    dataframe = pd.concat([dataframe, add_sumatra], ignore_index=True)
    return dataframe
