import numpy as np
from other_functions import haversine

def filter_column_data(earthquake_db):
    cols_to_filter = [col for col in earthquake_db.columns if "Err" in col]

    for col in cols_to_filter:
        print(col,'old nan value :',earthquake_db[col].isna().sum())

        #for each column we compute the median and standard deviation. we chose median because it's more 'typical' value
        median = earthquake_db[col].median()
        std = earthquake_db[col].std()

        #we create our bounds
        lower_bound = median - 3 * std
        upper_bound = median + 3 * std

        #we filter the rows based on the lower and upper bounds we just created, only for non-NaN values
        mask = (earthquake_db[col].isna() | ((earthquake_db[col] >= lower_bound) & (earthquake_db[col] <= upper_bound)))
        earthquake_db = earthquake_db[mask].copy()

        #we consider nan values as no mistake. Deliberate choice.
        earthquake_db[col].fillna(0, inplace=True)

        print(f"median error in {col} : {median}")
        print(col,'new nan value :',earthquake_db[col].isna().sum())
        print('-----------------------------------')

    earthquake_db = earthquake_db.dropna(subset=['latitude', 'longitude', 'depth', 'mag']) # We NEED all these values, we can't allow approximation here
    earthquake_db = earthquake_db.drop(cols_to_filter, axis = 1)
    return earthquake_db

def filter_earthquake_database(earthquake_db, threshold_year = 1995, min_mag = 5, depth_drop_range = [120,450]):
    #note taht we dont use the depth filter anymore but can still be used.

    reduced_db = (
        earthquake_db
        .copy()
        .loc[
            (earthquake_db['time'] >= f'{threshold_year}-01-01') #&
            # (earthquake_db['depth'] < depth_drop_range[0]) |
            # (earthquake_db['depth'] > depth_drop_range[1])
        ]
    )

    big_earthquakes = (
        reduced_db
        .copy()
        .loc[(reduced_db['mag'] >= min_mag)]
    )

    min_mag = big_earthquakes['mag'].min()
    max_mag = big_earthquakes['mag'].max()
    big_earthquakes['normalized_mag'] = (big_earthquakes['mag'] - min_mag) / (max_mag - min_mag)


    small_earthquakes = (
        reduced_db
        .copy()
        .loc[(reduced_db['mag'] < min_mag)]
    )
    return reduced_db, big_earthquakes, small_earthquakes, min_mag, max_mag

def clean_db_data(earthquake_db,date_threshold,mag_outlier_threshold,min_mag):
    print(f"filtering earthquake database for earthquakes after {date_threshold} and ensure absence of mistakes in database by dropping earthquakes of mag < {mag_outlier_threshold}...\n")
    earthquake_db = earthquake_db[(earthquake_db["date"] > date_threshold) & (earthquake_db["mag"] >= mag_outlier_threshold)]
    print(f"filtering 'X_Error' columns. If an 'X_Error' value is not a NaN and outside 3 sigma, we drop the row (inaccurate data). NaN values are replaced by median:\n")
    earthquake_db = filter_column_data(earthquake_db)

    earthquake_db['BigShock'] = earthquake_db['mag'] >= min_mag
    return earthquake_db
