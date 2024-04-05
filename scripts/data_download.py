"""
Functions to download USGS data by trimester since 1973 (current year - 53 years)
"""

import requests
import os

from tqdm import tqdm
from datetime import datetime

from datetime import datetime, timedelta

def list_years(prev_years, current_year):

    start_year = current_year - prev_years
    end_year = current_year
    return list(range(start_year, end_year + 1))


def generate_url(start_date, end_date):
    return f"https://earthquake.usgs.gov/fdsnws/event/1/query.csv?starttime={start_date}&endtime={end_date}&minmagnitude=2.5&eventtype=earthquake&orderby=time-asc"


def get_current_date():
    datetime_current_date = datetime.now()
    current_year = datetime_current_date.year
    current_date = datetime_current_date.strftime("%Y-%m-%d")
    return current_year, current_date


def splitURL(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str.replace("%20", " "),  "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(end_date_str.replace("%20", " "),  "%Y-%m-%d %H:%M:%S")

    middle_date = start_date + (end_date - start_date) / 2

    first_group_end = (middle_date - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
    second_group_start = (middle_date + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")

    return [start_date_str,first_group_end], [second_group_start,end_date_str]


def download_files(year,current_year,data_folder_path):

    #https://earthquake.usgs.gov/ only allows up to 20.000 data per query. it appears that this number can be reached in one
    #year or even one semester (i.e. second semester part of 2018). To deal with that we split the time range in 2 evertyime its needed
    #avoid server error. (3rd trimester of 2018 has more than 18 000 values alone)

    ## CANT RETRIEVE IF A FILE HAS BEEN DELETED OR NOT AFTER P0
    i = 0
    start_date, end_date = f"{year}-01-01%2000:00:00",f"{year}-12-31%2023:59:59"
    url = generate_url(start_date, end_date)
    filename = f"earthquakes_{year}_P0.csv"
    filepath = os.path.join(data_folder_path, filename)
    all_files= os.listdir(data_folder_path)
    if all((f"{year}_P0.csv") not in file_name for file_name in all_files) or year == current_year:
    # if not os.path.exists(f"earthquakes_{year}_P.csv"):
        if all((f"{year}_P1.csv") not in file_name for file_name in all_files) or year == current_year: #check if we already downloaded the year or not
            response = requests.get(url)
            if response.status_code == 200:
                with open(filepath, 'wb') as file:
                    file.write(response.content)
                print(f'Year {year}, part {i} -> file downloaded')

            else:
                print(f'Year {year} -> file too large... splitting into 2')
                semester1, semester2 = splitURL(start_date, end_date)
                for semester_range in [semester1, semester2]:

                    url = generate_url(semester_range[0], semester_range[1])
                    filename = f"earthquakes_{year}_P{i}.csv"
                    filepath = os.path.join(data_folder_path, filename)
                    if not os.path.exists(filepath) or year == current_year:
                        response = requests.get(url)
                        if response.status_code == 200:
                            i+=1
                            with open(filepath, 'wb') as file:
                                file.write(response.content)
                            print(f'Year {year}, part {i} -> file downloaded')
                        else:
                            print(f'Year {year}, semester -> file too large... splitting into 2')
                            trimester1, trimester2 = splitURL(semester_range[0], semester_range[1])
                            for trimester in [trimester1, trimester2]:
                                url = generate_url(trimester[0], trimester[1])
                                filename = f"earthquakes_{year}_P{i}.csv"
                                filepath = os.path.join(data_folder_path, filename)
                                if not os.path.exists(filepath) or year == current_year:
                                    response = requests.get(url)
                                    if response.status_code == 200:
                                        i+=1
                                        with open(filepath, 'wb') as file:
                                            file.write(response.content)
                                        print(f'Year {year}, trimester -> file downloaded')
                                    else:
                                        print(f">> WARNING : there is an issue with year {year}")


def download_data(prev_years,folder_path = "../ressources/usgs"):

    current_year, current_date = get_current_date()
    data_folder_path = os.path.join(folder_path)

    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)

    years_list = list_years(prev_years,current_year)

    for year in tqdm(years_list[:], desc="Downloading data"):

        download_files(year,current_year,data_folder_path)

    # remove_duplicate(folder_path, "Tdate")
    return
