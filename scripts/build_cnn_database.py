import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from other_functions import get_mainshock


def get_cnn_db_folder(current_directory,metric):
    """
    Get the path where we will save the arrays to, and load the arrays from.
    Create the path if not exists (or have been deleted)
    """
    arrays_for_cnn_folder_path = os.path.join(current_directory,f'../ressources/cnn/{metric}')
    if not os.path.exists(arrays_for_cnn_folder_path):
        os.makedirs(arrays_for_cnn_folder_path)

    return arrays_for_cnn_folder_path


def clean_cnn_db_folder(arrays_for_cnn_folder_path):
    """
    Function create with chat GPT

    Clean the folder of its previous file (in case you changed one parameter, to avoid having duplicates for example)
    """
    if os.path.exists(arrays_for_cnn_folder_path):
        for file_name in os.listdir(arrays_for_cnn_folder_path):
            file_path = os.path.join(arrays_for_cnn_folder_path, file_name)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        print(f"All files inside {arrays_for_cnn_folder_path} have been removed.")
    else:
        print(f"The folder {arrays_for_cnn_folder_path} does not exist.")


def get_all_clusters(clusters_df, as_only = True):

    if as_only: #if we only work on aftershocks
        # Returns dataframe with mainshock and aftershocks !!
        dataframe = clusters_df.copy()[clusters_df['delta_days']>=0.].reset_index(drop=True)
    else:
        # Returns whole dataset !!
        dataframe = clusters_df.copy()

    #We retrieve the different clusters
    grouped_dataframe = dataframe.groupby(['MC_lab','SC_lab', 'ST_lab'])
    print(f"we will work on {len(dataframe)} quakes spread over {len(grouped_dataframe)} clusters")
    return dataframe, grouped_dataframe


def count_max_intensity(dataframe,grouped_dataframe,n_days,metric, as_only=True):
    """
    Determine the maximum intensity over all our arrays. The value will be used to normalize all the arrays between 0 and 1.
    Returns the max intensity and the maximum value of an aftershock (or mainshock if not only aftershocks)
    """
    max_mag_as = 0

    n_days_df = dataframe.copy()[dataframe['delta_days']<=n_days]

    min_mag = dataframe.mag.min()
    max_mag = dataframe.mag.max()

    n_bins = int(10*(max_mag-min_mag))

    mag_bin_range = np.linspace(min_mag,max_mag, n_bins)
    score_bin_range = np.linspace(0,n_days_df[metric].max(), int(len(mag_bin_range)*2))

    max_counts = 0

    for _, cluster_data_df in grouped_dataframe:
        cluster_data = cluster_data_df.copy()[cluster_data_df['delta_days'] <= n_days].reset_index(drop=True)

        if as_only: #to avoid taking mainshock
            max_magnitude_index = cluster_data.mag.idxmax()
            cluster_data = cluster_data[max_magnitude_index+1:].reset_index(drop=True)

        max_mag_as = cluster_data.mag.max() if cluster_data.mag.max()>max_mag_as else max_mag_as

        if 75<= len(cluster_data):  #after experiments. Plus, we know we need several aftershocks to get redudancy in bins

            counts, _, _ = np.histogram2d(
                x=cluster_data['mag'],
                y=cluster_data[metric],
                bins=[mag_bin_range, score_bin_range]
            )

            max_counts = counts.max() if counts.max()>max_counts else max_counts

    return max_counts,max_mag_as


def build_database_metric(arrays_for_cnn_folder_path, grouped_dataframe, n_days, metric, max_counts, mag_bin_range, score_bin_range, as_only = True):
    """
    Builds our array database in function of the metrics and other parameters.
    """
    nb_files = 0

    for (mc_lab, sc_lab, st_lab), cluster_data_df in grouped_dataframe:
        cluster_data = cluster_data_df.copy()[cluster_data_df['delta_days'] <= n_days].reset_index(drop=True)

        out_of_range_data = cluster_data_df.copy()[cluster_data_df['delta_days'] > n_days].reset_index(drop=True)
        as_after_ndays = len(out_of_range_data)

        mainshock = get_mainshock(cluster_data)
        ms_mag = mainshock.mag

        if as_only: #only aftershocks, no mainshock
            max_magnitude_index = cluster_data.mag.idxmax()
            cluster_data = cluster_data[max_magnitude_index+1:].reset_index(drop=True)

        if len(cluster_data)>0:
            # ms_date = mainshock.time
            counts, magnitudes, scores = np.histogram2d(
                x=cluster_data['mag'],                      #we don't add the mainshock data to increase the difficulty (and avoid overfitting)
                y=cluster_data[metric],                     #we don't add the mainshock data to increase the difficulty (and avoid overfitting)
                bins=[mag_bin_range, score_bin_range]
            )

            counts = counts
            normalized_counts = counts / max_counts
            np.savez(f'{arrays_for_cnn_folder_path}/{mc_lab}_{sc_lab}_{st_lab}_signature__{as_after_ndays}__{ms_mag}.npz', counts=normalized_counts, mags=magnitudes, scores=scores)
            nb_files +=1

        # if ms_mag > 9:
        #     fig, ax = plt.subplots(1, 1, figsize=(9,9))

        #     im = ax.imshow(normalized_counts.T, extent=[4, 8.8, scores.min(), scores.max()],
        #                             origin='lower', aspect='auto', cmap='Greys_r')# , vmax=1.0)
        #     ax.set_xlabel('Magnitude')
        #     ax.set_ylabel('Score')


        #     plt.tight_layout()
        #     plt.show()

    print(f'Database for metric : {metric} created. Contains {nb_files} files.')
