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

valid_statin = valid_station(0, 60, states)

    #

state_data = {}

distribution_data = {}

for state_name in states:

    # set the directory name
    parent_dir = onedrive_path + state_name

    state_data[state_name] = {}

    # read each file and get data and generate the sdf curve
    for jj in range(len(valid_statin)):

        stat_all = np.zeros([4, 4])

        state_data[state_name][jj] = {}

        # get the station name from the file names

        csv_files = glob.glob(os.path.join(parent_dir, "*.csv"))

        csv_files.remove(parent_dir + '\station information.csv')

        temp_index = [i for i, s in enumerate(csv_files) if valid_statin.iloc[jj, 0] in s]

        raw_df = pd.read_csv(csv_files[temp_index[0]], encoding='unicode_escape')

        raw_df['Datetime'] = pd.to_datetime(raw_df['Datetime'])

        raw_df = raw_df.loc[(raw_df.iloc[:, 0] < '2021-1-1')]

        station_name = valid_statin.iloc[jj, 0]

        for duration in range(2, 3):

            year_min = raw_df.iloc[:, 0].dt.year.max()

            temp = raw_df.loc[(raw_df.iloc[:, 0].dt.year >= year_min - 50) & (raw_df.iloc[:, 0].dt.year <= year_min)]

            all_sdf = pydat.sdf_creator(data=temp, figure=False)[0]

            all_sdf = all_sdf['Duration='+str(duration)]

            all_sdf = all_sdf.sort_values(by='Severity(%)')

            all_distribution_params = stats.genextreme.fit(all_sdf.iloc[:, 3].dropna() * -1)

            all_distribution_cdf = stats.genextreme.cdf(np.linspace(0, 50, num=100), *all_distribution_params)

            plt.plot(np.linspace(0, 50, num=100), all_distribution_cdf, label='all')

            zz = 0

            for j in [10, 20, 30]:

                year_min = raw_df.iloc[:, 0].dt.year.max()

                data_part = raw_df.loc[(raw_df.iloc[:, 0].dt.year >= year_min - j - 50) & (raw_df.iloc[:, 0].dt.year <= year_min - j)]

                part_sdf = pydat.sdf_creator(data=data_part, figure=False)[0]

                part_sdf = part_sdf['Duration='+str(duration)]

                part_sdf = part_sdf.sort_values(by='Severity(%)')

                part_distribution_params = stats.genextreme.fit(part_sdf.iloc[:, 3].dropna() * -1)

                part_distribution_cdf = stats.genextreme.cdf(np.linspace(0, 50, num=100), *part_distribution_params)

                plt.plot(np.linspace(0, 50, num=100), part_distribution_cdf, label='part'+str(j))


                _, stat_all[zz, 0] = brunnermunzel(part_distribution_cdf, all_distribution_cdf)

                _, stat_all[zz, 1] = kruskal(part_distribution_cdf, all_distribution_cdf)

                res = cramervonmises_2samp(part_distribution_cdf, all_distribution_cdf)

                df_bins = pd.DataFrame()
                _, bins = pd.qcut(part_distribution_cdf, q=7, retbins=True)
                df_bins['bin'] = pd.cut(part_distribution_cdf, bins=bins).value_counts().index
                # Apply bins to both groups
                df_bins['part'] = pd.cut(part_distribution_cdf, bins=bins).value_counts().values
                df_bins['all'] = pd.cut(all_distribution_cdf, bins=bins).value_counts().values
                # Compute expected frequency in the treatment group
                df_bins['all_expected'] = df_bins['part'] / np.sum(df_bins['part']) * np.sum(df_bins['all'])

                _, stat_all[zz, 3] = chisquare(df_bins['all'], df_bins['all_expected'])

                #stat_all[zz, 2] = res.pvalue
                print(scipy.special.kl_div(part_distribution_cdf, all_distribution_cdf))
                stat_all[zz, 2] = sum(scipy.special.kl_div(all_distribution_cdf, part_distribution_cdf))

                zz += 1








            plt.xlabel('Wind')
            plt.ylabel('Cumulative probability')
            plt.legend()
            plt.show()

        state_data[state_name][jj] = stat_all



print(scipy.special.rel_entr( [0.2, 0.2, 0.4, 0.6, 0.8], [0.1, 0.2, 0.3, 0.4, 0.5]))

