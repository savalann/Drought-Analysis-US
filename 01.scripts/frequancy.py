#%% importing the libraries

import numpy as np
import datetime
import pandas as pd
import glob
import os
from dateutil.relativedelta import relativedelta
import scipy.stats as stats
import platform
from pydat import pydat
from station_statics import statistics
import matplotlib.pyplot as plt
from scipy.stats import brunnermunzel, cramervonmises_2samp, kstest, kruskal, chisquare
from scipy.stats import wasserstein_distance
import scipy


# %% platform detection and address assignment


if platform.system() == 'Windows':

    onedrive_path = 'E:/OneDrive/OneDrive - The University of Alabama/10.material/01.data/usgs_data/'

    box_path = 'C:/Users/snaserneisary/Box/Evaluation/Data_1980-2020/NWIS_sites/'

elif platform.system() == 'Darwin':

    onedrive_path = '/Users/savalan/Library/CloudStorage/OneDrive-TheUniversityofAlabama/10.material/01.data/usgs_data/'

states = ['al']


# %% functions


def valid_station(missing_value, data_number, states_list):

    station_list = statistics(states_list)

    states_list = 'al'

    modified_station = station_list[states_list][(station_list[states_list]['year_number'] >= data_number) &
                                                 (station_list[states_list]['missing_value_percent'] <= missing_value)]
    return modified_station


#%%

valid_statin_data = valid_station(5, 70, states)

print(len(valid_statin_data))

valid_statin_numbers = valid_statin_data.iloc[:, 0]

state_data = {}

distribution_data = {}

all_max = []

max_temp = 0

for state_name in states:

    # set the directory name
    parent_dir = onedrive_path + state_name

    state_data[state_name] = {}

    station_all = []

    # read each file and get data and generate the sdf curve
    for duration in range(2, 3):

        stat_all = np.zeros([len(valid_statin_numbers), 30])
        for station_index, station_number in enumerate(valid_statin_numbers):

            station_all.append(station_number)

            state_data[state_name][station_number] = {}

            # get the station name from the file names

            csv_files = glob.glob(os.path.join(parent_dir, "*.csv"))

            #csv_files.remove(parent_dir + '\station information.csv')

            temp_index = [i for i, s in enumerate(csv_files) if station_number in s]

            raw_df = pd.read_csv(csv_files[temp_index[0]], encoding='unicode_escape')

            raw_df['Datetime'] = pd.to_datetime(raw_df['Datetime'])

            raw_df = raw_df.loc[(raw_df.iloc[:, 0] < '2021-1-1')]

            station_name = station_number

            zz = 0

            all_sdf = pydat.sdf_creator(data=raw_df, figure=False)[0]

            all_sdf = all_sdf['Duration=' + str(duration)]

            max_temp = (all_sdf.iloc[:, 3] * -1).mean()

            all_max.append(max_temp)

            #if all_max < max_temp:
               # all_max = max_temp


        stat_all = np.zeros([len(valid_statin_numbers), 30])
        for station_index, station_number in enumerate(valid_statin_numbers):


            state_data[state_name][station_number] = {}

            # get the station name from the file names

            csv_files = glob.glob(os.path.join(parent_dir, "*.csv"))

            # csv_files.remove(parent_dir + '\station information.csv')

            temp_index = [i for i, s in enumerate(csv_files) if station_number in s]

            raw_df = pd.read_csv(csv_files[temp_index[0]], encoding='unicode_escape')

            raw_df['Datetime'] = pd.to_datetime(raw_df['Datetime'])

            raw_df = raw_df.loc[(raw_df.iloc[:, 0] < '2021-1-1')]

            station_name = station_number

            zz = 0
            for rp_index, return_period in enumerate([10, 20, 50, 100]):

                for jj, j in enumerate([0, 20, 40]):

                    year_max = raw_df.iloc[:, 0].dt.year.max()

                    data_part = raw_df.loc[(raw_df.iloc[:, 0].dt.year >= year_max - j - 30) & (raw_df.iloc[:, 0].dt.year <= year_max - j)]

                    part_sdf = pydat.sdf_creator(data=data_part, figure=False)[0]

                    part_sdf = part_sdf['Duration='+str(duration)]

                    part_sdf = part_sdf.sort_values(by='Severity(%)')

                    part_distribution_params = stats.genpareto.fit(part_sdf.iloc[:, 3].dropna() * -1)

                    part_distribution_cdf = stats.genpareto.cdf(part_sdf.iloc[:, 3].dropna() * -1, *part_distribution_params)



                    stat_all[station_index, zz+jj] = 1/(1-stats.genpareto.cdf(all_max[station_index], *part_distribution_params))



                zz += 7


print(stats.genpareto.cdf(all_max[station_index], *part_distribution_params))


station_info = pd.read_csv(onedrive_path + '/al/error_stations/station information.csv')

station_coordinate = station_info[station_info.site_no.isin(valid_statin_numbers.values.astype(int))].iloc[:,
                     1:6].reset_index(drop=True)







all_10 = pd.DataFrame({'station_number': station_all, '0':stat_all[:, 0] , '20':stat_all[:, 1]  , '40':stat_all[:, 2]  }).sort_values(by='station_number').reset_index(drop=True)
all_10['lat'] = station_coordinate.iloc[:, 3]
all_10['lon'] = station_coordinate.iloc[:, 4]
all_20 = pd.DataFrame({'station_number': station_all, '0':stat_all[:, 7] , '20':stat_all[:, 8]  , '40':stat_all[:, 9]  }).sort_values(by='station_number').reset_index(drop=True)
all_20['lat'] = station_coordinate.iloc[:, 3]
all_20['lon'] = station_coordinate.iloc[:, 4]
all_50 = pd.DataFrame({'station_number': station_all, '0':stat_all[:, 14] , '20':stat_all[:, 15]  , '40':stat_all[:, 16]  }).sort_values(by='station_number').reset_index(drop=True)
all_50['lat'] = station_coordinate.iloc[:, 3]
all_50['lon'] = station_coordinate.iloc[:, 4]
all_100 = pd.DataFrame({'station_number': station_all, '0':stat_all[:, 21] , '20':stat_all[:, 22]  , '40':stat_all[:, 23]  }).sort_values(by='station_number').reset_index(drop=True)
all_100['lat'] = station_coordinate.iloc[:, 3]
all_100['lon'] = station_coordinate.iloc[:, 4]

sheet_name = [10, 20, 50, 100]

writer = pd.ExcelWriter(state_name+'_frequency_test.xlsx', engine='xlsxwriter')
for jj, state_name in enumerate([all_10, all_20, all_50, all_100]):


    state_name.to_excel(writer, sheet_name=str(sheet_name[jj]))
writer.save()
writer.close()

#%% Figure

# set width of bar
zz = [10, 20, 50, 100]
for jj, state_name in enumerate([all_10, all_20, all_50, all_100]):
    barWidth = 0.25
    fig = plt.subplots(dpi=300)

    # set height of bar
    IT = state_name.iloc[3:6, 1]
    ECE = state_name.iloc[3:6, 2]
    CSE = state_name.iloc[3:6, 3]

    # Set position of bar on X axis
    br1 = np.arange(len(IT))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, IT, color='r', width=barWidth,
            edgecolor='grey', label='2020-1990')
    plt.bar(br2, ECE, color='g', width=barWidth,
            edgecolor='grey', label='2000-1970')
    plt.bar(br3, CSE, color='b', width=barWidth,
            edgecolor='grey', label='1980-1950')

    # Adding Xticks
    plt.xlabel('Station Number', fontweight='bold', fontsize=15)
    plt.ylabel('Severity', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(IT))],
               state_name.iloc[3:6, 0])

    plt.legend()
    plt.savefig(str(zz[jj])+'_years_return_period_frequancy_test')




