import numpy as np

def select_big_earthquakes(earthquake_db, n_earthquakes):
    """
    Args :
        - earthquake_db : our earthquakes database (dataframe)
        - n_earthquakes : the number of earthquakes we want to select

    Return:
        - biggest_earthquakes : a dataframe containing the biggest earthquakes of the database, ranked by magnitude

    """
    biggest_earthquakes = earthquake_db.nlargest(n_earthquakes, 'mag')
    biggest_earthquakes = biggest_earthquakes[['time','latitude', 'longitude', 'mag', 'depth', 'nearest_area']].reset_index(drop = True)

    return biggest_earthquakes


def valid_year(data_files, fyear):
    """
    Args :
        - data_files : the list of filenames
        - fyear : the threshold year we want to start using the data


    Return:
        - fyear : the threshold year we can and/or want to start using the data

    This function verifies if the first and last year set by the user are not bigger or smaller
    than the first/last year of the selected datafiles. If it's the case, then it updates the
    value of the first and last year for the rest of the program
    """
    data_first_year = int(data_files[0].split('_')[1])

    if fyear < data_first_year:
        fyear = data_first_year

    return fyear


def haversine(lat1, lon1, lat2, lon2):
    """
    Args :
        - lat1 : latitude of first location
        - lat2 : latitude of second location
        - lon1 : longitude of first location
        - lon2 : longitude of second location

    Return:
        - distance : distance between the 2 locations in kilometers

    pseudo code : https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/
    """
    #radius of the Earth in kilometers
    radius = 6378.1

    #converts the latitudes and longitudes into radians angles
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    #computes the distance for longitude and latitude (in rad)
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    #applying haversine :
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = radius * c

    return distance


def grid_cell(step):
    #step is used as the length of the cells in the grid
    min_latitude,max_latitude,min_longitude,max_longitude = -90, 90, - 180, 180
    #these are the global limits for the earth

    num_latitude_cells = int((max_latitude - min_latitude) / step)
    num_longitude_cells = int((max_longitude - min_longitude) / step)

    #initialization of the grid
    grid = np.zeros((num_latitude_cells, num_longitude_cells, 4))

    for i in range(num_latitude_cells):
        for j in range(num_longitude_cells):
            #we fill in the grid with the lat and long of each cell
            grid[i, j, 0] = min_latitude + i * step
            grid[i, j, 1] = min_longitude + j * step
            grid[i, j, 2] = min_latitude + (i + 1) * step
            grid[i, j, 3] = min_longitude + (j + 1) * step

    return grid

def create_earth_grid(power_scale = 4, step_num = 7):
    """
    We want to use cells of square shape in order to simplify the maintenance.
    Using cells of different latiude/longitude ratio would lead to ajacent-cells
    to form a rectangle block instead of a square (which is closer to a circle)
    """

    step =  step_num/(2**power_scale)
    grid = grid_cell(step= step)

    return grid, step

def get_grid_cell(lat, lon, step):
    min_latitude, min_longitude= -90,-180

    lat_index = ((lat - min_latitude) / step).astype(int)
    lon_index = ((lon - min_longitude) / step).astype(int)
    grid_cell = list(zip(lat_index, lon_index))

    return grid_cell

def get_cell_coordinates(grid,i,j):

    min_lat = grid[i, j, 0]
    min_lon = grid[i, j, 1]
    max_lat = grid[i, j, 2]
    max_lon = grid[i, j, 3]

    return min_lat,min_lon,max_lat,max_lon


def bayes_rule(probA, probB, probBA):
    """
        Calculate the bayes rule with the given probability.

        Args:
            probA (double) : Correspond to the probability of an evenment A has occured.
            probB (double) : Correspond to the probability of an evenment B has occured.
            probBA (double) : Correspond to the probability of an evenment B has occured if an evenment A has occured.

        Returns:
            double : This return the probability of an evenment A given the evenment B has occured.

        Raises:
            No raise exception in this function.

        Examples:
            >>> bayes_rule(0.12, 0.34, 0.84)
             Result : 0.29647058823529404

        Note:
            Parameters have to be between 0 and 1 inclusive.
    """
    # if 0.0 <= probA <= 1.0 or 0.0 <= probB <= 1.0 or 0.0 <= probBA <= 1.0:
    #     raise Exception("One of the probabilities given is not between 0 and 1.")
    result = probA * probBA / probB
    return result
def get_mainshock(cluster_data):
    return cluster_data.loc[cluster_data['mag'].idxmax()]
