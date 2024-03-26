"""
# Analysis of Reservoir Impact on Droughts

## Author
Name: Savalan Naser Neisary
Email: savalan.neisary@gmail.com | snaserneisary@crimson.ua.edu
Affiliation: PhD Student, University of Alabama
GitHub: savalann

## Code Description
Creation Date: 2023-03-25
This script is used to analyze reservoir impact on Droughts.

## License
This software is licensed under the Apache License 2.0. See the LICENSE file for more details.
"""

"""
1. Time scopes that I need to use: a)Seasonal b)annual 
2. The comparison metrics for comparing upstream and downstream stations:
    a) Number of drought events
    b) Correlation between stations
    c) Trend of drought events
    d) Plot their time series
    e) Compare the dominant season on annual drought









"""

# %% savvy: Import packages.

# savvy: basic packages.
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import pymannkendall as mk

# savvy: system packages


# savvy: system packages.
import os
import warnings

warnings.filterwarnings("ignore")

# savvy: my packages.
from pyndat import Pyndat

# %%
path_01 = os.getcwd()  # savvy: Get the current working directory
input_path_general = path_01.replace('01.scripts', '02.inputs/')  # savvy: set the path for input data.
input_path = path_01.replace('01.scripts', '03.outputs/')  # savvy: Set the path for output data.


# %% savvy: Seasonal function

def sdf_seasonal(data):
    quarterly_average = data.resample('Q').mean()
    quarterly_average['season'] = quarterly_average.index.quarter

    threshold = np.zeros(4)
    severity = quarterly_average.copy()
    for season_name in range(1, 5):
        temp_df = quarterly_average
        temp_df = temp_df[temp_df.season == season_name].USGS_flow.mean()
        threshold[season_name - 1] = temp_df
        severity.loc[severity['season'] == season_name, 'USGS_flow'] = (
            np.round((severity.loc[severity['season'] == season_name, 'USGS_flow'] - temp_df) / temp_df * 100, 0))

    severity_all = severity

    threshold_all = threshold

    return severity_all, threshold_all


# %%
station_upstream = None
station_downstream = None
duration_list = [ii for ii in range(2, 11)]
dataset_raw = pd.read_csv(f'{input_path_general}storage.csv', dtype='object')
end_year = 2020
data_length = 41
start_year = end_year - data_length + 1

# Main code that will be the function in the future.
data_station_raw = {'upstream': {}, 'downstream': {}}
drought_data = {'upstream': {}, 'downstream': {}}

for station_index in range(len(dataset_raw)):
    for station_location in ['upstream', 'downstream']:
        if station_location == 'upstream':
            station_upstream = dataset_raw.iloc[station_index, 1:4].dropna().tolist()
            if len(station_upstream) > 1:
                temp_df_01 = pd.read_csv(f'{input_path}usgs_data/{station_upstream[0]}.csv')[['Datetime', 'USGS_flow']]
                temp_df_01['Datetime'] = pd.to_datetime(temp_df_01['Datetime'], format='mixed')
                for station_num in station_upstream[1:]:
                    temp_df_02 = pd.read_csv(f'{input_path}usgs_data/{station_num}.csv')[['Datetime', 'USGS_flow']]
                    temp_df_02['Datetime'] = pd.to_datetime(temp_df_02['Datetime'], format='mixed')
                    temp_df_01 = pd.merge(temp_df_01, temp_df_02, on='Datetime')
                temp_df_01['USGS_flow'] = temp_df_01.iloc[:, 1:].sum(axis=1)
                temp_df_01 = temp_df_01[['Datetime', 'USGS_flow']]
            else:
                temp_df_01 = pd.read_csv(f'{input_path}usgs_data/{station_upstream[0]}.csv')
                temp_df_01['Datetime'] = pd.to_datetime(temp_df_01['Datetime'], format='mixed')
            data_station_raw['upstream'] = temp_df_01[(temp_df_01.Datetime >= f'{start_year}-10-01') &
                                                      (temp_df_01.Datetime < f'{end_year + 1}-11-01')]
            data_station_raw['upstream'].set_index('Datetime', inplace=True)
            drought_data['upstream'][dataset_raw.iloc[station_index]['Reservoir']] = (
                sdf_seasonal(data=data_station_raw['upstream'].iloc[:, :1]))[0]
        else:
            station_downstream = dataset_raw.iloc[station_index]['Downstream Station']
            temp_df_03 = pd.read_csv(f'{input_path}usgs_data/{station_downstream}.csv')
            temp_df_03['Datetime'] = pd.to_datetime(temp_df_03['Datetime'], format='mixed')
            data_station_raw['downstream'] = temp_df_03[(temp_df_03.Datetime >= f'{start_year}-10-01') &
                                                        (temp_df_03.Datetime < f'{end_year + 1}-11-01')]
            data_station_raw['downstream'].set_index('Datetime', inplace=True)
            drought_data['downstream'][dataset_raw.iloc[station_index]['Reservoir']] = (
                sdf_seasonal(data=data_station_raw['downstream'].iloc[:, :1]))[0]


# %% Drought data comparison analysis function

#
# p_correlation = np.zeros((len(dataset_raw), 1))
# score_correlation = np.zeros((len(dataset_raw), 1))
# p_trend = []
# slope_trend = []
# drought_number = []
# drought_descriptive = []
#
# for station_index in range(len(dataset_raw)):
#     temp_p_trend = []
#     temp_slope_trend = []
#     temp_drought_number = []
#     temp_drought_descriptive = []
#
#     data_upstream = (drought_data['upstream'][dataset_raw.iloc[station_index]
#     ['Reservoir']]).dropna()
#     data_upstream.rename(columns={'USGS_flow': 'Severity(%)'}, inplace=True)
#     data_downstream = (drought_data['downstream'][dataset_raw.iloc[station_index]
#     ['Reservoir']]).dropna()
#     data_downstream.rename(columns={'USGS_flow': 'Severity(%)'}, inplace=True)
#
#     temp_data_merged = pd.merge(data_upstream.reset_index(), data_downstream.reset_index(), on='Datetime')
#     temp_data_merged = temp_data_merged[temp_data_merged['season_x'] == 3]
#     score_correlation[station_index, 0], p_correlation[station_index, 0] = (
#         spearmanr(temp_data_merged['Severity(%)_x'], temp_data_merged['Severity(%)_y']))
#
#     temp_p_trend.append(mk.original_test(temp_data_merged['Severity(%)_x'])[0])
#     temp_p_trend.append(mk.original_test(temp_data_merged['Severity(%)_y'])[0])
#
#     temp_slope_trend.append(round(mk.original_test(temp_data_merged['Severity(%)_x'])[7], 2))
#     temp_slope_trend.append(round(mk.original_test(temp_data_merged['Severity(%)_y'])[7], 2))
#
#     temp_drought_number.append((temp_data_merged['Severity(%)_x'] < 0).sum())
#     temp_drought_number.append((temp_data_merged['Severity(%)_y'] < 0).sum())
#
#     temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]
#                                           ['Severity(%)_x'].max(), 2))
#     temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_y'] < 0]
#                                           ['Severity(%)_y'].max(), 2))
#
#     temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]
#                                           ['Severity(%)_x'].median(), 2))
#     temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_y'] < 0]
#                                           ['Severity(%)_y'].median(), 2))
#
#     temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]
#                                           ['Severity(%)_x'].min(), 2))
#     temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_y'] < 0]
#                                           ['Severity(%)_y'].min(), 2))
#
#     p_trend.extend([temp_p_trend])
#     slope_trend.extend([temp_slope_trend])
#     drought_number.extend([temp_drought_number])
#     drought_descriptive.extend([temp_drought_descriptive])
#
# index_1 = []
# index_2 = []
# index_3 = []
#
#
# index_3.append(f'correlation')
# for name_col in ['Upstream', 'Downstream']:
#     index_1.append(f'{name_col}')
#
# for name_metric in ['min', 'median', 'max']:
#     for name_col in ['Upstream', 'Downstream']:
#         index_2.append(f'{name_metric}_{name_col}')
#
# drought_number = pd.DataFrame(drought_number, columns=index_1)
# final_drought_number = pd.concat([dataset_raw, drought_number], axis=1)
#
# drought_descriptive = pd.DataFrame(drought_descriptive, columns=index_2) * -1
# final_drought_descriptive = pd.concat([dataset_raw, drought_descriptive], axis=1)
#
# p_trend = pd.DataFrame(p_trend, columns=index_1)
# final_p_trend = pd.concat([dataset_raw, p_trend], axis=1)
#
# slope_trend = pd.DataFrame(slope_trend, columns=index_1)
# final_slope_trend = pd.concat([dataset_raw, slope_trend], axis=1)
#
# score_correlation = pd.DataFrame(np.round(score_correlation, 2), columns=index_3)
# final_score_correlation = pd.concat([dataset_raw, score_correlation], axis=1)
#
# p_correlation = pd.DataFrame(np.round(p_correlation, 2), columns=index_3)
# p_correlation_text = p_correlation.applymap(lambda x: 'significant' if x <= 0.05 else ' not significant')
# final_p_correlation = pd.concat([dataset_raw, p_correlation_text], axis=1)
# # #
# # # final_drought_number.to_csv(f'{path_01}number_of_drought_events.csv')
# # # final_drought_descriptive.to_csv(f'{path_01}descriptive_info.csv')
# # # final_p_trend.to_csv(f'{path_01}trend_p_value.csv')
# # # final_slope_trend.to_csv(f'{path_01}trend_slope.csv')
# # # final_score_correlation.to_csv(f'{path_01}correlation_score.csv')
# # # final_p_correlation.to_csv(f'{path_01}correlation_p_value.csv')

# %%


season_list = [ii for ii in range(1, 5)]

data = drought_data['downstream'][dataset_raw.iloc[station_index]['Reservoir']]


final_data = {}

for duration_num in duration_list:
    final_data[duration_num] = {}
    temp_data_01 = data[data['season'] == 1]
    temp_data_02 = temp_data_01.rolling(window=duration_num).mean().dropna()
    temp_data_02 = temp_data_02.reset_index()
    temp_data_02['year'] = temp_data_02.Datetime.dt.year
    temp_data_02 = temp_data_02[['year', 'USGS_flow']]
    for season_num in season_list[1:]:
        temp_data_01 = data[data['season'] == season_num]
        temp_data_01 = temp_data_01.rolling(window=duration_num).mean().dropna()
        temp_data_01 = temp_data_01.reset_index()
        temp_data_01['year'] = temp_data_01.Datetime.dt.year
        temp_data_02 = pd.merge(temp_data_02, temp_data_01[['USGS_flow', 'year']], on='year', suffixes=(f'_{season_num-1}', f'_{season_num}')).dropna()
    final_data[duration_num] = temp_data_02
