"""
# Drought Analysis US

## Author
Name: Savalan Naser Neisary
Email: savalan.neisary@gmail.com | snaserneisary@crimson.ua.edu
Affiliation: PhD Student, University of Alabama
GitHub: savalann

## Code Description
Creation Date: 2023-02-23
This code downloads the USGS streamflow data for different states under a specific folder structure.
## License
This software is licensed under the Apache License 2.0. See the LICENSE file for more details.
"""

# %% savy: Import basic libraries
from pyndat import Pyndat
import datetime
import os


# %% savalan: Functions

def usgs_download(state, path):
    start = datetime.datetime.now()
    print('Run started')
    parent_dir = path

    for i in range(len(state)):

        error_stations = []

        directory = state[i]

        path_1 = os.path.join(parent_dir, directory)

        if not os.path.exists(path_1):
            os.makedirs(path_1)

        path_2 = os.path.join(path_1, 'error_stations')

        if not os.path.exists(path_2):
            os.makedirs(path_2)

        raw_data = Pyndat.valid_station(status='all', state=state[i])

        raw_data = raw_data[raw_data['stat_cd'] == 3]

        raw_data = raw_data[raw_data['site_tp_cd'] == 'ST']

        dup_list = raw_data[raw_data.duplicated(subset=['site_no'])]['site_no'].tolist()

        modified_data = raw_data['site_no']

        for j in range(len(modified_data)):

            print(j, '/', len(modified_data))

            data_raw_streamflow = Pyndat.daily_data(site=str(modified_data.iloc[j]))
            print(data_raw_streamflow.shape)
            print(modified_data.iloc[j])

            if len(data_raw_streamflow) != 0 and data_raw_streamflow.shape[1] == 6:

                min_year = data_raw_streamflow.Datetime.dt.year.min()

                max_year = data_raw_streamflow.Datetime.dt.year.max()

                if modified_data.iloc[j] not in dup_list:
                    path_writing = f'{path}{state[i]}/'

                if modified_data.iloc[j] in dup_list:
                    path_writing = f'{path}{state[i]}/error_stations/'

                sheet = str(modified_data.iloc[j]) + "_" + str(min_year) + '_' + str(max_year)

                data_raw_streamflow.to_csv(path_writing + sheet + ".csv", index=False, encoding="cp1252")

            else:

                error_stations.append(modified_data.iloc[j])

        path_writing = f'{path}{state[i]}/'

        raw_data.to_csv(path_writing + 'station information' + ".csv", index=False)

        total_stations = len(raw_data) - len(dup_list)

        duplicate_stations = len(dup_list)

        empty_stations = len(error_stations)

        if os.path.exists(path_writing + "info.txt") is False:

            f = open(path_writing + "info.txt", "x")

            f.write("Creation of first files on " + datetime.datetime.today().strftime('%d %B %Y'))

        else:

            f = open(path_writing + "info.txt", "r")

            lines = len(f.readlines())

            f = open(path_writing + "info.txt", "a")

            f.write("\nUpdate " + str(lines - 1) + ' on ' + datetime.datetime.today().strftime('%d %B %Y'))

        f.write("\nTotal number of stations = " + str(total_stations))

        f.write("\nTotal number of duplicate stations = " + str(duplicate_stations))

        f.write("\nTotal number of empty stations = " + str(empty_stations))

        f.write("\nThese stations had no data = " + ", ".join(error_stations))

        f.write("\nThese stations had duplicates = " + ", ".join(dup_list))

        f.write("\n=================================================================================================")

        f.close()

        print(state[i], 'finished')

        print(state[i], 'Run Time:', str(datetime.datetime.now() - start))

    print('Total Run Time:', str(datetime.datetime.now() - start))










