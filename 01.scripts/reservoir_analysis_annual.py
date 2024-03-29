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
    f) Compare the correlation between seasons and annual. 








"""

# %% savvy: Import packages.

# savvy: basic packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# savvy: statistical packages
from scipy.stats import spearmanr, kendalltau
import pymannkendall as mk

# savvy: system packages.
import math
import warnings
import os
warnings.filterwarnings("ignore")

# savvy: my packages.
from pyndat import Pyndat

# %% savvy: Drought data generation function.
path_01 = os.getcwd()  # savvy: Get the current working directory
input_path_general = path_01.replace('01.scripts', '02.inputs/')  # savvy: set the path for input data.
input_path = path_01.replace('01.scripts', '03.outputs/')  # savvy: Set the path for output data.
# Temp variables - Inputs.

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

# %% Drought data comparison analysis function

temp_result = np.zeros((len(dataset_raw), 9))
p_correlation = np.zeros((len(dataset_raw), 9))
score_correlation = np.zeros((len(dataset_raw), 9))
p_trend = []
slope_trend = []
drought_number = []
drought_descriptive = []

for station_index in range(len(dataset_raw)):
    temp_p_trend = []
    temp_slope_trend = []
    temp_drought_number = []
    temp_drought_descriptive = []
    for duration_number in duration_list:

        data_upstream = (drought_data['upstream'][dataset_raw.iloc[station_index]
        ['Reservoir']][f'Duration={duration_number}']).dropna()
        data_downstream = (drought_data['downstream'][dataset_raw.iloc[station_index]
        ['Reservoir']][f'Duration={duration_number}']).dropna()

        temp_data_merged = pd.merge(data_upstream, data_downstream, on='Date')
        score_correlation[station_index, duration_number - 2], p_correlation[station_index, duration_number - 2] = (
            spearmanr(temp_data_merged['Severity(%)_x'], temp_data_merged['Severity(%)_y']))

        temp_p_trend.append(mk.original_test(temp_data_merged['Severity(%)_x'])[0])
        temp_p_trend.append(mk.original_test(temp_data_merged['Severity(%)_y'])[0])

        temp_slope_trend.append(round(mk.original_test(temp_data_merged['Severity(%)_x'])[7], 2))
        temp_slope_trend.append(round(mk.original_test(temp_data_merged['Severity(%)_y'])[7], 2))

        temp_drought_number.append((temp_data_merged['Severity(%)_x'] < 0).sum())
        temp_drought_number.append((temp_data_merged['Severity(%)_y'] < 0).sum())

        temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]
                                              ['Severity(%)_x'].max(), 2))
        temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_y'] < 0]
                                              ['Severity(%)_y'].max(), 2))

        temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]
                                              ['Severity(%)_x'].median(), 2))
        temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_y'] < 0]
                                              ['Severity(%)_y'].median(), 2))

        temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_x'] < 0]
                                              ['Severity(%)_x'].min(), 2))
        temp_drought_descriptive.append(round(temp_data_merged[temp_data_merged['Severity(%)_y'] < 0]
                                              ['Severity(%)_y'].min(), 2))

    p_trend.extend([temp_p_trend])
    slope_trend.extend([temp_slope_trend])
    drought_number.extend([temp_drought_number])
    drought_descriptive.extend([temp_drought_descriptive])

index_1 = []
index_2 = []
index_3 = []

for ii in range(2, 11):
    index_3.append(f'D{ii}')
    for name_col in ['Upstream', 'Downstream']:
        index_1.append(f'D{ii}_{name_col}')

    for name_metric in ['min', 'median', 'max']:
        for name_col in ['Upstream', 'Downstream']:
            index_2.append(f'D{ii}_{name_metric}_{name_col}')

drought_number = pd.DataFrame(drought_number, columns=index_1)
final_drought_number = pd.concat([dataset_raw, drought_number], axis=1)

drought_descriptive = pd.DataFrame(drought_descriptive, columns=index_2) * -1
final_drought_descriptive = pd.concat([dataset_raw, drought_descriptive], axis=1)

p_trend = pd.DataFrame(p_trend, columns=index_1)
final_p_trend = pd.concat([dataset_raw, p_trend], axis=1)

slope_trend = pd.DataFrame(slope_trend, columns=index_1)
final_slope_trend = pd.concat([dataset_raw, slope_trend], axis=1)

score_correlation = pd.DataFrame(np.round(score_correlation, 2), columns=index_3)
final_score_correlation = pd.concat([dataset_raw, score_correlation], axis=1)

p_correlation = pd.DataFrame(np.round(p_correlation, 2), columns=index_3)
p_correlation_text = p_correlation.applymap(lambda x: 'significant' if x <= 0.05 else ' not significant')
final_p_correlation = pd.concat([dataset_raw, p_correlation_text], axis=1)

# # final_drought_number.to_csv(f'{path_01}number_of_drought_events.csv')
# # final_drought_descriptive.to_csv(f'{path_01}descriptive_info.csv')
final_p_trend.to_csv(f'{input_path}annual_trend_p_value.csv')
final_slope_trend.to_csv(f'{input_path}annual_trend_slope.csv')
# # final_score_correlation.to_csv(f'{path_01}correlation_score.csv')
# # final_p_correlation.to_csv(f'{path_01}correlation_p_value.csv')

# %%



def create_line_plot(datasets, save_path, figsize=(12, 12)):
    n_subplots = len(datasets)
    n_cols = int(math.ceil(math.sqrt(n_subplots)))
    n_rows = int(math.ceil(n_subplots / n_cols))
    key_name = list(datasets.keys())
    # Using sharey=True and sharex=True
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=True, dpi=300)
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, ax in enumerate(axes):
        if i < n_subplots:
            # Plotting the data with labels for the legend
            # Setting the positions and width for the bars
            positions = np.arange(1, len(datasets[key_name[i]])+1)  # Row numbers
            width = 0.35  # The width of the bars

            # Plotting the bars for each dataset
            ax.bar(positions, datasets[key_name[i]].iloc[:, 2], width, label='Downstream')
            ax.bar(positions, datasets[key_name[i]].iloc[:, 1], width, label='Upstream')


            ax.set_title(f'{datasets[key_name[i]].iloc[0, -1]} ({datasets[key_name[i]].iloc[0, -2]})')
            ax.legend()
            # Setting the x-axis label for the last row
            if i // n_cols == n_rows - 1:
                ax.set_xlabel('Year')

            # Setting the y-axis label for the first column
            if i % n_cols == 0:
                ax.set_ylabel('Severity(%)')
        else:
            # Hide unused subplots
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_path}bar_annual_severity.png')
    plt.show()

#%%
import numpy as np
from scipy import stats

def statisical_test(datasets):
    # Sample data - replace these with your actual datasets
    data1 = datasets.iloc[:, 1]
    data2 = datasets.iloc[:, 2]

    # Calculate the differences
    differences = data2 - data1

    # Normality test on the differences
    stat, p_value = stats.shapiro(differences)
    print('Shapiro-Wilk Test: stat=%.3f, p=%.3f' % (stat, p_value))

    # Check if differences are normally distributed
    # if p_value > 0.05:
    #     print("Differences are normally distributed, performing paired t-test.")
    #     # Perform paired t-test
    #     t_stat, p_val = stats.ttest_rel(data1, data2)
    #     print('Paired t-test: t=%.3f, p=%.3f' % (t_stat, p_val))
    # else:
    print("Differences are not normally distributed, performing Wilcoxon signed-rank test.")
    # Perform Wilcoxon signed-rank test
    w_stat, p_val = stats.wilcoxon(data1, data2)
    print('Wilcoxon signed-rank test: W=%.3f, p=%.3f' % (w_stat, p_val))


#%%


temp_data_merged = {}

for duration_number in [2]:
    for reservoir_number in range(len(dataset_raw)):
        temp_data_upstream = drought_data['upstream'][dataset_raw.iloc[reservoir_number]['Reservoir']]
        temp_data_upstream = temp_data_upstream[f'Duration={duration_number}'][['Date', 'Severity(%)']]
        temp_data_downstream = drought_data['downstream'][dataset_raw.iloc[reservoir_number]['Reservoir']]
        temp_data_downstream = temp_data_downstream[f'Duration={duration_number}'][['Date', 'Severity(%)']]
        temp_data_merged_01 = pd.merge(temp_data_upstream, temp_data_downstream, on='Date',
                                       suffixes=('_upstream', '_downstream'))
        print(duration_number, reservoir_number, '=======================')
        statisical_test(temp_data_merged_01.dropna())
        temp_data_merged_01['Capacity'] = dataset_raw.iloc[reservoir_number]['Capacity']
        temp_data_merged_01['Reservoir'] = dataset_raw.iloc[reservoir_number]['Reservoir']
        temp_data_merged[dataset_raw.iloc[reservoir_number]['Reservoir']] = temp_data_merged_01

    # create_line_plot(temp_data_merged, f'{input_path}d{duration_number}_')



#%%
#
# def create_scatter_plot(datasets, save_path, figsize=(12, 12)):
#     n_subplots = len(datasets)
#     n_cols = int(math.ceil(math.sqrt(n_subplots)))
#     n_rows = int(math.ceil(n_subplots / n_cols))
#     key_name = list(datasets.keys())
#     # Using sharey=True and sharex=True
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=True, dpi=300)
#     axes = axes.flatten()  # Flatten the axes array for easy iteration
#
#     for i, ax in enumerate(axes):
#         if i < n_subplots:
#             # Plotting the data with labels for the legend
#             # Setting the positions and width for the bars
#             positions = np.arange(1, len(datasets[key_name[i]])+1)  # Row numbers
#             width = 0.35  # The width of the bars
#
#             # Plotting the bars for each dataset
#             ax.scatter(datasets[key_name[i]].iloc[:, 2], datasets[key_name[i]].iloc[:, 3])
#             ax.plot([10, 30], [10, 30], 'k--')  # This plots the diagonal line
#
#             ax.set_title(f'Duration = {i+2}')
#             # Setting the x-axis label for the last row
#             if i // n_cols == n_rows - 1:
#                 ax.set_xlabel('Upstream Drought Number')
#
#             # Setting the y-axis label for the first column
#             if i % n_cols == 0:
#                 ax.set_ylabel('Downstream Drought Number')
#         else:
#             # Hide unused subplots
#             ax.axis('off')
#
#     plt.tight_layout()
#     plt.savefig(f'{save_path}scatter_annual_drought_number.png')
#     plt.show()
#
# temp_data_plot = {}
# for duration_number in duration_list:
#
#     temp_data_plot[duration_number] = final_drought_number[['Reservoir', 'Capacity', f'D{duration_number}_Upstream', f'D{duration_number}_Downstream']]
#
# create_scatter_plot(temp_data_plot, f'{input_path}d{2}_')
#
#
#
#
#
#
#
#
