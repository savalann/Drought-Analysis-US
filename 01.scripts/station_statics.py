"""
# Drought Analysis US

## Author
Name: savy Naser Neisary
Email: savy.neisary@gmail.com | snaserneisary@crimson.ua.edu
Affiliation: PhD Student, University of Alabama
GitHub: savyn

## Code Description
Creation Date: 2023-02-23
This code adds number of years, missing value, and the ending year of time series according to the downloaded file.

## License
This software is licensed under the Apache License 2.0. See the LICENSE file for more details.
"""

# %% savy: Import libraries
import numpy as np
import datetime
import pandas as pd
import glob
import os
from datetime import datetime

# %% savy: Functions
def statistics(state_name, start_year, end_year, path):
    # savy: Get the name of streamflow data files.
    parent_dir = path + state_name
    csv_files = glob.glob(os.path.join(parent_dir, "*.csv"))

    # savy: Make an empty list and dictionary for the requested state and stations.
    station_name_list = []
    station_stat_list = np.zeros([len(csv_files), 3])

    # savy: Get time series of each station and calculate the statistics.
    for file_index, files in enumerate(csv_files):
        raw_df = pd.read_csv(files, encoding='unicode_escape')
        raw_df['Datetime'] = pd.to_datetime(raw_df['Datetime'], format='mixed')
        raw_df = raw_df[(raw_df['Datetime'].dt.year >= start_year) & (raw_df['Datetime'].dt.year <= end_year)]

        # savy: Extract the station number out of the file name.
        station_name_list.append(files.replace(parent_dir, '')[1:-14])
        if len(raw_df) > 0:
            # savy: Get number of years
            station_stat_list[file_index, 0] = len(raw_df['Datetime'].dt.year.drop_duplicates())

            # savy: Get the missing data percentage.
            station_stat_list[file_index, 1] = abs(
                np.round(((datetime(end_year, 12, 31) - datetime(start_year, 1, 1)).days - len(raw_df)) /
                         (datetime(end_year, 12, 31) - datetime(start_year, 1, 1)).days * 100, 0))

            # savy: Get the last year of the time series.
            station_stat_list[file_index, 2] = str(raw_df.iloc[-1, 0].year)

        else:
            station_stat_list[file_index, 0] = np.nan
            station_stat_list[file_index, 1] = np.nan
            station_stat_list[file_index, 2] = np.nan

    # Make a dataframe and dictionary for the output data.
    state_statistics = pd.DataFrame({'station': station_name_list, 'year_number': station_stat_list[:, 0],
                                     'missing_value_percent': station_stat_list[:, 1],
                                     'last_year': station_stat_list[:, 2]})
    return state_statistics


def study_area_stations(state_name, start, end, huc_2, path):
    parent_dir = path + state_name
    station_info_raw = pd.read_csv(parent_dir + '/error_stations/station information.csv')
    station_stat = statistics(state_name, start, end, path)
    station_stat_temp = station_info_raw[station_info_raw.site_no.isin(station_stat.station.apply(np.int64))] \
        .reset_index(drop=True)
    station_stat_temp = station_stat_temp.rename(columns={'site_no': 'station'})
    station_stat_temp.pop('agency_cd')
    station_stat_temp['state'] = state_name
    station_stat_temp['end_date'] = pd.to_datetime(station_stat_temp['end_date'], format='mixed')
    station_stat['station'] = station_stat['station'].astype(np.int64, copy=False)
    station_stat_temp = station_stat_temp.merge(station_stat.reset_index(drop=True), on='station')
    station_info_modified = station_stat_temp[(station_stat_temp['huc_cd'] >= huc_2*1e6) &
                                              (station_stat_temp['huc_cd'] < (huc_2+1)*1e6) &
                                              (station_stat_temp['end_date'].dt.year >= end)].reset_index(drop=True)
    station_info_modified = station_info_modified.dropna(subset=['last_year'])
    #
    return station_info_modified


