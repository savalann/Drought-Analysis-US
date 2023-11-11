import numpy as np
import datetime
import pandas as pd
import glob
import os

from dateutil.relativedelta import relativedelta
import platform


# %% platform detection and address assignment


if platform.system() == 'Windows':

    onedrive_path = 'E:/OneDrive/OneDrive - The University of Alabama/10.material/01.data/usgs_data/'

    box_path = 'C:/Users/snaserneisary/Box/Evaluation/Data_1980-2020/NWIS_sites/'

elif platform.system() == 'Darwin':

    onedrive_path = '/Users/savalan/Library/CloudStorage/OneDrive-TheUniversityofAlabama/10.material/01.data/usgs_data/'



# %% functions
def statistics(state_name):
    # Get the name of streamflow data files.
    parent_dir = onedrive_path + state_name
    csv_files = glob.glob(os.path.join(parent_dir, "*.csv"))

    # Make an empty list and dictionary for the requested state and stations.
    station_name_list = []
    station_stat_list = np.zeros([len(csv_files), 2])

    # Get time series of each station and calculate the statistics.
    for file_index, files in enumerate(csv_files):
        raw_df = pd.read_csv(files, encoding='unicode_escape')
        raw_df['Datetime'] = pd.to_datetime(raw_df['Datetime'])

        # Extract the station number out of the file name.
        station_name_list.append(files.replace(parent_dir, '')[1:-14])

        # Get number of years
        station_stat_list[file_index, 0] = len((raw_df['Datetime'].dt.year).drop_duplicates())

        # Get the missing data percentage.
        station_stat_list[file_index, 1] = abs(np.round(((raw_df.iloc[-1, 0] - raw_df.iloc[0, 0]).days - len(raw_df)) /
                                                        (raw_df.iloc[-1, 0] - raw_df.iloc[0, 0]).days * 100, 0))

    # Make a dataframe and dictionary for the output data.
    state_statistics = pd.DataFrame({'station': station_name_list, 'year_number': station_stat_list[:, 0],
                                                 'missing_value_percent': station_stat_list[:, 1]})

    return state_statistics



# PREVIOUS VERSION
'''def statistics(states):

    state_stat = {}

    for state_name in states:

        parent_dir = onedrive_path + state_name

        csv_files = glob.glob(os.path.join(parent_dir, "*.csv"))

        #csv_files.remove(parent_dir + '\station information.csv')

        station_name_list = []

        station_stat_list = np.zeros([len(csv_files), 2])

        for jj, files in enumerate(csv_files):

            raw_df = pd.read_csv(files, encoding='unicode_escape')

            raw_df['Datetime'] = pd.to_datetime(raw_df['Datetime'])

            station_name_list.append(files.replace(parent_dir, '')[1:-14])

            station_stat_list[jj, 0] = len((raw_df['Datetime'].dt.year).drop_duplicates())

            station_stat_list[jj, 1] = abs(np.round(((raw_df.iloc[-1, 0] - raw_df.iloc[0, 0]).days - len(raw_df)) /
                                                (raw_df.iloc[-1, 0] - raw_df.iloc[0, 0]).days * 100, 0))

        final_df = pd.DataFrame({'station': station_name_list, 'year_number': station_stat_list[:, 0],
                                 'missing_value_percent': station_stat_list[:, 1]})

        state_stat[state_name] = final_df

    return state_stat'''