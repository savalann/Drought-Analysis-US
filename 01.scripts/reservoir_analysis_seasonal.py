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

# savvy: system packages.
import warnings

warnings.filterwarnings("ignore")

# savvy: my packages.
from pyndat import Pyndat


#%%



# %% savvy: Drought data generation function.

# Temp variables - Inputs.
input_path = ('E:/OneDrive/OneDrive - The University of Alabama/02.projects/02.nidis/02.code/Drought-Analysis-US'
              '/03.outputs/')
station_upstream = None
station_downstream = None
duration_list = [ii for ii in range(2, 11)]
dataset_raw = pd.read_csv(f'{input_path}storage.csv', dtype='object')
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
            data_station_raw['upstream'] = temp_df_01[(temp_df_01.Datetime >= f'{start_year}-01-01') &
                                                      (temp_df_01.Datetime < f'{end_year + 1}-01-01')]
            drought_data['upstream'][dataset_raw.iloc[station_index]['Reservoir']] = (
                Pyndat.sdf_creator(data=data_station_raw['upstream'], figure=False))[3]
        else:
            station_downstream = dataset_raw.iloc[station_index]['Downstream Station']
            temp_df_03 = pd.read_csv(f'{input_path}usgs_data/{station_downstream}.csv')
            data_station_raw['downstream'] = temp_df_03[(temp_df_03.Datetime >= f'{start_year}-01-01') &
                                                        (temp_df_03.Datetime < f'{end_year + 1}-01-01')]
            drought_data['downstream'][dataset_raw.iloc[station_index]['Reservoir']] = (
                Pyndat.sdf_creator(data=data_station_raw['downstream'], figure=False))[3]

# # %% Drought data comparison analysis function
#
# temp_result = np.zeros((len(dataset_raw), 9))
# p_correlation = np.zeros((len(dataset_raw), 9))
# score_correlation = np.zeros((len(dataset_raw), 9))
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
#     for duration_number in duration_list:
#
#         data_upstream = (drought_data['upstream'][dataset_raw.iloc[station_index]
#         ['Reservoir']][f'Duration={duration_number}']).dropna()
#         data_downstream = (drought_data['downstream'][dataset_raw.iloc[station_index]
#         ['Reservoir']][f'Duration={duration_number}']).dropna()
#
#         temp_data_merged = pd.merge(data_upstream, data_downstream, on='Date')
#         score_correlation[station_index, duration_number - 2], p_correlation[station_index, duration_number - 2] = (
#             spearmanr(temp_data_merged['Severity(%)_x'], temp_data_merged['Severity(%)_y']))
#
#         temp_p_trend.append(mk.original_test(temp_data_merged['Severity(%)_x'])[0])
#         temp_p_trend.append(mk.original_test(temp_data_merged['Severity(%)_y'])[0])
#
#         temp_slope_trend.append(round(mk.original_test(temp_data_merged['Severity(%)_x'])[7], 2))
#         temp_slope_trend.append(round(mk.original_test(temp_data_merged['Severity(%)_y'])[7], 2))
#
#         temp_drought_number.append((temp_data_merged['Severity(%)_x'] < 0).sum())
#         temp_drought_number.append((temp_data_merged['Severity(%)_y'] < 0).sum())
#
#         temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]
#                                               ['Severity(%)_x'].max(), 2))
#         temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_y'] < 0]
#                                               ['Severity(%)_y'].max(), 2))
#
#         temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]
#                                               ['Severity(%)_x'].median(), 2))
#         temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_y'] < 0]
#                                               ['Severity(%)_y'].median(), 2))
#
#         temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]
#                                               ['Severity(%)_x'].min(), 2))
#         temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_y'] < 0]
#                                               ['Severity(%)_y'].min(), 2))
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
# for ii in range(2, 11):
#     index_3.append(f'D{ii}')
#     for name_col in ['Upstream', 'Downstream']:
#         index_1.append(f'D{ii}_{name_col}')
#
#     for name_metric in ['min', 'median', 'max']:
#         for name_col in ['Upstream', 'Downstream']:
#             index_2.append(f'D{ii}_{name_metric}_{name_col}')
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
#
# # # final_drought_number.to_csv(f'{path_01}number_of_drought_events.csv')
# # # final_drought_descriptive.to_csv(f'{path_01}descriptive_info.csv')
# # # final_p_trend.to_csv(f'{path_01}trend_p_value.csv')
# # # final_slope_trend.to_csv(f'{path_01}trend_slope.csv')
# # # final_score_correlation.to_csv(f'{path_01}correlation_score.csv')
# # # final_p_correlation.to_csv(f'{path_01}correlation_p_value.csv')


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
dd = data_station_raw['upstream']
dd['Datetime'] = pd.to_datetime(dd['Datetime'])
dd_2 = dd.set_index('Datetime')
aa = sdf_seasonal(dd_2.iloc[:, :1])[0]








