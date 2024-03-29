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
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# savvy: system packages.
import os
import warnings
import math
warnings.filterwarnings("ignore")

# savvy: my packages.
from pyndat import Pyndat

# %%
path_01 = os.getcwd()  # savvy: Get the current working directory
input_path_general = path_01.replace('01.scripts', '02.inputs/')  # savvy: set the path for input data.
input_path = path_01.replace('01.scripts', '03.outputs/')  # savvy: Set the path for output data.


# %% savvy: Seasonal function

def seasonal_severity(data):
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


def seasonal_sdf(data):
    data = seasonal_severity(data)[0]
    season_list = [ii for ii in range(1, 5)]
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
            temp_data_02 = pd.merge(temp_data_02, temp_data_01[['USGS_flow', 'year']], on='year',
                                    suffixes=(f'_{season_num - 1}', f'_{season_num}')).dropna()
        final_data[duration_num] = temp_data_02

    return final_data


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
        drought_data[station_location][dataset_raw.iloc[station_index]['Reservoir']] = {}
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
            drought_data['upstream'][dataset_raw.iloc[station_index]['Reservoir']]['annual'] = (
                Pyndat.sdf_creator(data=data_station_raw['upstream'], figure=False))[3]
            data_station_raw['upstream'].set_index('Datetime', inplace=True)
            drought_data['upstream'][dataset_raw.iloc[station_index]['Reservoir']]['seasonal'] = (
                seasonal_sdf(data=data_station_raw['upstream'].iloc[:, :1]))
        else:
            station_downstream = dataset_raw.iloc[station_index]['Downstream Station']
            temp_df_03 = pd.read_csv(f'{input_path}usgs_data/{station_downstream}.csv')
            temp_df_03['Datetime'] = pd.to_datetime(temp_df_03['Datetime'], format='mixed')
            data_station_raw['downstream'] = temp_df_03[(temp_df_03.Datetime >= f'{start_year}-10-01') &
                                                        (temp_df_03.Datetime < f'{end_year + 1}-11-01')]
            drought_data['downstream'][dataset_raw.iloc[station_index]['Reservoir']]['annual'] = (
                Pyndat.sdf_creator(data=data_station_raw['downstream'], figure=False))[3]
            data_station_raw['downstream'].set_index('Datetime', inplace=True)
            drought_data['downstream'][dataset_raw.iloc[station_index]['Reservoir']]['seasonal'] = (
                seasonal_sdf(data=data_station_raw['downstream'].iloc[:, :1]))


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

#%% MAIN FUNCTION

p_correlation_up = np.zeros((len(dataset_raw), 4))
score_correlation_up = np.zeros((len(dataset_raw), 4))
p_correlation_down = np.zeros((len(dataset_raw), 4))
score_correlation_down = np.zeros((len(dataset_raw), 4))

final_p_up_all = {}
final_score_up_all = {}
final_p_down_all = {}
final_score_down_all = {}

for duration_num in duration_list:
    for station_index in range(len(dataset_raw)):
        temp_data_upstream_annual = (drought_data['upstream'][dataset_raw.iloc[station_index]['Reservoir']]['annual']
                                     [f'Duration={duration_num}'][['Date', 'Severity(%)']])
        temp_data_downstream_annual = (drought_data['downstream'][dataset_raw.iloc[station_index]['Reservoir']]
                                       ['annual'][f'Duration={duration_num}'][['Date', 'Severity(%)']])
        temp_data_upstream_annual.rename(columns={'Date': 'year'}, inplace=True)
        temp_data_downstream_annual.rename(columns={'Date': 'year'}, inplace=True)
        temp_data_upstream_seasonal = (
            drought_data)['upstream'][dataset_raw.iloc[station_index]['Reservoir']]['seasonal'][duration_num]
        temp_data_downstream_seasonal = (
            drought_data)['downstream'][dataset_raw.iloc[station_index]['Reservoir']]['seasonal'][duration_num]

        temp_data_merged_upstream = pd.merge(temp_data_upstream_annual, temp_data_upstream_seasonal, on='year')
        temp_data_merged_downstream = pd.merge(temp_data_downstream_annual, temp_data_downstream_seasonal, on='year')

        for season_num in range(1, 5):
            p_correlation_up[station_index, season_num-1] = spearmanr(
                temp_data_merged_upstream['Severity(%)'], temp_data_merged_upstream[f'USGS_flow_{season_num}'])[1]

            score_correlation_up[station_index, season_num-1] = spearmanr(
                temp_data_merged_upstream['Severity(%)'], temp_data_merged_upstream[f'USGS_flow_{season_num}'])[0]

            p_correlation_down[station_index, season_num-1] = spearmanr(
                temp_data_merged_downstream['Severity(%)'], temp_data_merged_downstream[f'USGS_flow_{season_num}'])[1]

            score_correlation_down[station_index, season_num-1] = spearmanr(
                temp_data_merged_downstream['Severity(%)'], temp_data_merged_downstream[f'USGS_flow_{season_num}'])[0]

    p_correlation_up = np.array(p_correlation_up, dtype=float)
    significance = p_correlation_up <= 0.05
    p_correlation_up = np.where(significance, 'significant', 'not significant')
    p_correlation_down = np.array(p_correlation_down, dtype=float)
    significance = p_correlation_down <= 0.05
    p_correlation_down = np.where(significance, 'significant', 'not significant')

    final_p_up_all[duration_num] = p_correlation_up
    final_score_up_all[duration_num] = score_correlation_up
    final_p_down_all[duration_num] = p_correlation_down
    final_score_down_all[duration_num] = score_correlation_down
# %% Draw the bar plots
def create_bar_plot(datasets, save_path, figsize=(12, 12)):
    n_subplots = len(datasets)
    n_cols = int(math.ceil(math.sqrt(n_subplots)))
    n_rows = int(math.ceil(n_subplots / n_cols))
    #key_name = list(datasets.keys())
    # Using sharey=True and sharex=True
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=True, dpi=300)
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, ax in enumerate(axes):
        if i < n_subplots:
            # Plotting the data with labels for the legend
            # Setting the positions and width for the bars
            positions = np.arange(1, 5)  # Row numbers
            width = 0.35  # The width of the bars

            # Plotting the bars for each dataset
            ax.bar(positions - width / 2, datasets.iloc[i, 2:6], width, label='Upstream')
            ax.bar(positions + width / 2, datasets.iloc[i, 6:], width, label='Downstream')
            ax.set_title(f'{datasets.iloc[i, 0]} ({datasets.iloc[i, 1]})')
            ax.legend()
            # Setting the x-axis label for the last row
            if i // n_cols == n_rows - 1:
                ax.set_xlabel('Season Number')

            # Setting the y-axis label for the first column
            if i % n_cols == 0:
                ax.set_ylabel('Correlation')
        else:
            # Hide unused subplots
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_path}correlation_bar_plot.png')
    plt.show()


for duration_num in [2]:
    temp_dataset = np.concatenate((final_score_up_all[duration_num], final_score_down_all[duration_num]), axis=1)
    temp_dataset = pd.DataFrame(temp_dataset)
    dataset_raw['Capacity'] = dataset_raw['Capacity'].astype(float).astype('int64')
    temp_dataset = pd.concat([dataset_raw[['Reservoir', 'Capacity']], temp_dataset], axis=1)
    create_bar_plot(temp_dataset, f'{input_path}d{duration_num}_')



#%%


def create_line_plot(datasets, save_path, figsize=(12, 12)):
    n_subplots = len(datasets)
    n_cols = int(math.ceil(math.sqrt(n_subplots)))
    n_rows = int(math.ceil(n_subplots / n_cols))
    #key_name = list(datasets.keys())
    # Using sharey=True and sharex=True
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=True, dpi=300)
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, ax in enumerate(axes):
        if i < n_subplots:
            # Plotting the data with labels for the legend
            # Setting the positions and width for the bars
            positions = np.arange(1, 5)  # Row numbers
            width = 0.35  # The width of the bars

            # Plotting the bars for each dataset
            ax.bar(positions - width / 2, datasets.iloc[i, 2:6], width, label='Upstream')
            ax.bar(positions + width / 2, datasets.iloc[i, 6:], width, label='Downstream')
            ax.set_title(f'{datasets.iloc[i, -1]} storage:({datasets.iloc[i, -2]})')
            ax.legend()
            # Setting the x-axis label for the last row
            if i // n_cols == n_rows - 1:
                ax.set_xlabel('Season Number')

            # Setting the y-axis label for the first column
            if i % n_cols == 0:
                ax.set_ylabel('Correlation')
        else:
            # Hide unused subplots
            ax.axis('off')

    plt.tight_layout()
    #plt.savefig(f'{save_path}correlation_bar_plot.png')
    plt.show()

for duration_num in [2]:
    for station_index in range(len(dataset_raw)):
        temp_data_upstream_annual = (drought_data['upstream'][dataset_raw.iloc[station_index]['Reservoir']]['annual']
                                     [f'Duration={duration_num}'][['Date', 'Severity(%)']])
        temp_data_downstream_annual = (drought_data['downstream'][dataset_raw.iloc[station_index]['Reservoir']]
                                       ['annual'][f'Duration={duration_num}'][['Date', 'Severity(%)']])

        temp_dataset = pd.merge([temp_data_upstream_annual, temp_data_downstream_annual], on='Date')
        create_line_plot(temp_dataset, f'{input_path}d{duration_num}_')
