# %% importing the libraries

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
from mpl_toolkits.axisartist import Axes


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


# %%

valid_statin_data = valid_station(10, 40, states)

valid_statin_numbers = valid_statin_data.iloc[:, 0]

empirical_distribution_data = {}

distribution_data = {}

severity_data = {}

distribution = stats.genpareto

for state_name in states:

    # set the directory name
    parent_dir = onedrive_path + state_name

    severity_data[state_name] = {}

    empirical_distribution_data[state_name] = {}

    distribution_data[state_name] = {}

    csv_files = glob.glob(os.path.join(parent_dir, "*.csv"))

    #csv_files.remove(parent_dir + '/station information.csv')

    severity = np.zeros([len(valid_statin_numbers), 4])

    # read each file and get data and generate the sdf curve
    test = []
    for station_index, station_number in enumerate(valid_statin_numbers):

        test.append(station_number)

        # get the station name from the file names

        temp_index = [i for i, s in enumerate(csv_files) if station_number in s]

        raw_df = pd.read_csv(csv_files[temp_index[0]], encoding='unicode_escape')

        raw_df['Datetime'] = pd.to_datetime(raw_df['Datetime'])

        raw_df = raw_df.loc[(raw_df.iloc[:, 0] < '2021-1-1')]

        all_sdf = pydat.sdf_creator(data=raw_df, figure=False)[0]

        empirical_distribution_data[state_name][station_number] = all_sdf

        distribution_data[state_name][station_number] = {}

        for duration in range(2, 11):

            all_sdf_temp = all_sdf['Duration=' + str(duration)]

            all_sdf_temp = all_sdf_temp.sort_values(by='Severity(%)')

            all_distribution_params = distribution.fit(all_sdf_temp.iloc[:, 3].dropna() * -1)

            all_distribution_cdf = distribution.cdf(all_sdf_temp.iloc[:, 3].dropna() * -1, *all_distribution_params)

            distribution_data[state_name][station_number][duration] = pd.DataFrame({'severity': all_sdf_temp.iloc[:, 3].dropna() * -1, 'cdf': all_distribution_cdf})

        color = ['b', 'g', 'y', 'r', 'orange', 'brown', 'gray', 'cyan', 'olive', 'pink']
        fig = plt.figure(dpi=300, layout="constrained", facecolor='whitesmoke')
        axs = fig.add_subplot(axes_class=Axes, facecolor='whitesmoke')
        axs.axis["right"].set_visible(False)
        axs.axis["top"].set_visible(False)
        axs.axis["left"].set_axisline_style("-|>")
        axs.axis["left"].line.set_facecolor("black")
        axs.axis["bottom"].set_axisline_style("-|>")
        axs.axis["bottom"].line.set_facecolor("black")
        plt.title(label='SDF Curve', fontsize=20, pad=10)
        axs.axis["bottom"].label.set_text("Severity (%)")
        axs.axis["bottom"].label.set_fontsize(15)
        axs.axis["left"].label.set_text("Non-Exceedance Probability")
        axs.axis["left"].label.set_fontsize(15)
        for duration in range(2, 11):
            data = distribution_data[state_name][station_number][duration]
            filled_marker_style = dict(marker='o', linestyle='-', markersize=5,
                                       color=color[duration-2])
            axs.plot(data.iloc[:, 0], data.iloc[:, 1],  **filled_marker_style, label=('Duration = ' + str(duration)))
        axs.legend(loc='lower right')
        axs.set(xlabel='Severity (%)', ylabel='Non-Exceedance Probability (%)',
                title=state_name + ' - ' + station_number + ' - SDF Curves')
        plt.savefig('alwrc_result/sdf_curve/al/' + str(station_number))


    for duration in range(2, 11):
        for station_index, station_number in enumerate(valid_statin_numbers):
            all_sdf_temp_1 = empirical_distribution_data[state_name][station_number]

            all_sdf_temp_2 = all_sdf_temp_1['Duration=' + str(duration)]

            all_sdf_temp_2 = all_sdf_temp_2.sort_values(by='Severity(%)')

            all_distribution_params = distribution.fit(all_sdf_temp_2.iloc[:, 3].dropna() * -1)

            for rp_num, return_period in enumerate([10, 20, 50, 100]):
                severity[station_index, rp_num] = (distribution.ppf(1 - 1 / return_period, *all_distribution_params))

            temp_severity = pd.DataFrame({'station_name': valid_statin_numbers.reset_index
            (drop=True), '10 year': severity[:, 0], '20 year': severity[:, 1], '50 year': severity[:, 2], '100 year':
                                              severity[:, 3]})

            station_info = pd.read_csv(onedrive_path + '/al/error_stations/station information.csv')

            station_coordinate = station_info[station_info.site_no.isin(valid_statin_numbers.values.astype(int))].iloc[:,
                                 1:6].reset_index(drop=True)

            temp_severity = temp_severity.sort_values(by='station_name').reset_index(drop=True)

            temp_severity['lat'] = station_coordinate.iloc[:, 3]

            temp_severity['lon'] = station_coordinate.iloc[:, 4]

            severity_data[state_name][duration] = temp_severity

#%%

for state_name in states:
    writer = pd.ExcelWriter(state_name+'_1.xlsx', engine='xlsxwriter')
    for duration in range(2, 11):
        temp_data = severity_data[state_name][duration]
        temp_data.to_excel(writer, sheet_name=str(duration))
writer.save()
writer.close()

