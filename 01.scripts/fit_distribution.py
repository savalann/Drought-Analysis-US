# This script fits different distributions to different stations to find the best distribution. Its result is three
# metrics for goodness of fit.

# %% importing the libraries

# basic packages
import numpy as np
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# system packages
import glob
import os
import platform
from pathlib import Path

# analysis packages
import scipy.stats as stats

# my packages
from pydat import pydat
from station_statics import statistics
# %% platform detection and address assignment


if platform.system() == 'Windows':

    onedrive_path = 'E:/OneDrive/OneDrive - The University of Alabama/10.material/01.data/usgs_data/'

    box_path = 'C:/Users/snaserneisary/Box/Evaluation/Data_1980-2020/NWIS_sites/'

elif platform.system() == 'Darwin':

    onedrive_path = '/Users/savalan/Library/CloudStorage/OneDrive-TheUniversityofAlabama/10.material/01.data/usgs_data/'

    onedrive_path_new = '/Users/savalan/Library/CloudStorage/OneDrive-TheUniversityofAlabama/02.projects/02.nidis/06.results/01.paper/'

# basic parameter assignment
states = ['al', 'ak']
# 'al', 'ak', 'la', 'ms', 'tn', 'ky', 'ga', 'sc', 'nc', 'fl'
duration_list = [i for i in range(2, 11)]

# List of candidate distributions to check
distributions_all = [

    stats.pearson3,  # Pearson III
    stats.gennorm,  # Generalized Normal
    stats.genextreme,  # Generalized Extreme
    stats.genlogistic,  # Generalized Logistic
    stats.genpareto,  # Generalized Pareto
    stats.gamma,  # Gamma
    stats.beta,  # Beta
    stats.weibull_min,  # Weibull Minimum

    # Add more distributions here if needed
]

# %% functions

def valid_station(missing_value, data_number, states_list):
    station_list = statistics(states_list)
    states_list = states_list
    modified_station = station_list[(station_list['year_number'] >= data_number) &
                                                 (station_list['missing_value_percent'] <= missing_value)]
    return modified_station


def find_best_fit_distribution(data, distribution_list):
    distributions = distribution_list
    # Calculate the Kolmogorov-Smirnov test statistic for each distribution
    station_result = np.zeros([5, len(distributions)])
    distribution_name = []

    for ii, distribution in enumerate(distributions):
        # Fit the distribution to the data
        params = distribution.fit(data.iloc[:, 3].dropna() * -1)
        # Get the cumulative distribution function (CDF) for the fitted distribution
        cdf = distribution.cdf(data.iloc[:, 3].dropna() * -1, *params)
        # Calculate the Kolmogorov-Smirnov test statistic
        ks_statistic = stats.kstest(data.iloc[:, 3].dropna() * -1, distribution.cdf, args=params)[0]
        ks_p_value = stats.kstest(data.iloc[:, 3].dropna() * -1, distribution.cdf, args=params)[1]
        log_likelihood = distribution.logpdf(data.iloc[:, 3].dropna() * -1, *params).sum()
        p = len(params)
        n = len(data.iloc[:, 3].dropna())
        #if n/p < 40:
        #   aic = 2 * (p * n / (n - p - 1)) - 2 * log_likelihood
        #else:
        aic = 2 * p - 2 * log_likelihood
        bic = np.log(n) * p - 2 * log_likelihood

        wass_metric = stats.wasserstein_distance(cdf, data.iloc[:, 4].dropna())

        station_result[0, ii] = ks_statistic
        station_result[1, ii] = ks_p_value
        station_result[2, ii] = aic
        station_result[3, ii] = bic
        station_result[4, ii] = wass_metric
        distribution_name.append(distribution.name)

    station_result = pd.DataFrame({'distribution': distribution_name, 'statics': station_result[0, :],
                                   'p_value': station_result[1, :], 'aic': station_result[2, :],
                                   'bic': station_result[3, :], 'wass': station_result[4, :]})

    # station_result = station_result.sort_values(by='p_value', ascending=False) #

    return station_result


# def find_best_fit_distribution(data):


# %% run the functions

# create variables
state_data = {}
distribution_data = {}
valid_station_list = {}

for state_name in states:
    valid_statin = valid_station(0, 30, state_name)
    valid_station_list[state_name] = valid_statin

    # set the directory name
    parent_dir = onedrive_path + state_name

    # set variables
    state_data[state_name] = {}

    # read each file and get data and generate the sdf curve
    for jj in range(len(valid_statin)):
        # get the station name from the file names

        # files = os.get_files_in_directory(parent_dir + '/')

        csv_files = glob.glob(os.path.join(parent_dir, "*.csv"))

        # csv_files.remove(parent_dir + '\station information.csv')

        station_name = valid_statin.iloc[jj, 0]

        temp_index = [i for i, file_name in enumerate(csv_files) if station_name in file_name]

        raw_df = pd.read_csv(csv_files[temp_index[0]], encoding='unicode_escape')

        state_data[state_name][station_name] = pydat.sdf_creator(data=raw_df, figure=False)[0]

    distribution_data[state_name] = {}

    for duration in duration_list:

        distribution_data[state_name][duration] = {}

        for station in valid_statin['station']:
            data = state_data[state_name][station]['Duration=' + str(duration)]

            temp_my_variable = find_best_fit_distribution(data, distributions_all)

            distribution_data[state_name][duration][station] = temp_my_variable

# %% evaluate the fitness functions


distribution_stat_list = {}
best_p_value_list = {}
best_aic_list = {}
best_bic_list = {}
best_wass_list = {}

distribution_name = []

for distribution in distributions_all:
    distribution_name.append(distribution.name)

for state_name in states:
    best_p_value_list[state_name] = {}
    best_aic_list[state_name] = {}
    best_bic_list[state_name] = {}
    best_wass_list[state_name] = {}

    for duration in duration_list:
        valid_statin = valid_station_list[state_name]

        distribution_stat = np.zeros(len(distributions_all))
        for zz, station in enumerate(valid_statin['station']):
            distribution_data_temp = distribution_data[state_name][duration][station].reset_index(drop=True)
            for distribution_index, distribution in enumerate(distribution_data_temp.iloc[:, 0]):
                if distribution_data_temp.iloc[distribution_index, 2] < 0.05:
                    distribution_stat[distribution_index] += 1
        distribution_stat_temp = pd.DataFrame(
            {'distribution_name': distribution_name, 'station_number': distribution_stat})
        distribution_stat_list[state_name + '_duration_' + str(duration)] = distribution_stat_temp

    for duration in duration_list:

        station_best_result_p_value = []

        station_best_result_aic = []

        station_best_result_bic = []

        station_best_result_wass_metric = []

        for zz, station in enumerate(valid_statin['station']):
            distribution_data_temp = distribution_data[state_name][duration][station].reset_index(drop=True)
            if (distribution_stat_temp.iloc[:, 1] == 0).any():
                distribution_data[state_name][duration][station] = distribution_data_temp[
                    distribution_stat_temp.iloc[:, 1] < np.round(len(valid_statin) / 10, 0)]

            data_temp = distribution_data[state_name][duration][station]
            data_temp = data_temp.sort_values(by='p_value', ascending=False)
            station_best_result_p_value.append(data_temp.iloc[0, 0])

            data_temp = distribution_data[state_name][duration][station]
            data_temp = data_temp.sort_values(by='aic')
            station_best_result_aic.append(data_temp.iloc[0, 0])

            data_temp = distribution_data[state_name][duration][station]
            data_temp = data_temp.sort_values(by='bic')
            station_best_result_bic.append(data_temp.iloc[0, 0])

            data_temp = distribution_data[state_name][duration][station]
            data_temp = data_temp.sort_values(by='wass')
            station_best_result_wass_metric.append(data_temp.iloc[0, 0])

        best_p_value = np.round(pd.DataFrame(station_best_result_p_value).value_counts() / len(valid_statin) * 100, 0)
        best_aic = np.round(pd.DataFrame(station_best_result_aic).value_counts() / len(valid_statin) * 100, 0)
        best_bic = np.round(pd.DataFrame(station_best_result_bic).value_counts() / len(valid_statin) * 100, 0)
        best_wass = np.round(pd.DataFrame(station_best_result_wass_metric).value_counts() / len(valid_statin) * 100, 0)

        best_p_value_list[state_name][duration] = best_p_value
        best_aic_list[state_name][duration] = best_aic
        best_bic_list[state_name][duration] = best_bic
        best_wass_list[state_name][duration] = best_wass

        print(str(duration) + state_name + '====================================================')
        print(best_p_value)
        print('-----------------------------------------')
        print(best_aic)
        print('-----------------------------------------')
        print(best_bic)
        print('-----------------------------------------')
        print(best_wass)
        print('-----------------------------------------')

print('Finish')


# %% prepare the result folders

for state_name in states:
    newpath = onedrive_path_new + state_name + '/'
    Path(newpath).mkdir(parents=True, exist_ok=True)
    Path(newpath+'distribution_search_result/').mkdir(parents=True, exist_ok=True)




# %% print the outputs




