import dataretrieval.nwis as nwis
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.axisartist import Axes

class pydat:

    #  The function to give the station names for each state or basin.
    def valid_station(status='optimal', state='', basin='', start_date="1800-01-01",
                      end_date=datetime.today().strftime('%Y-%m-%d')):

        sufficient_year_number = [30]  # The least number of years which is sufficient for SDF curve creation.

        parameter_code = "00060"  # The parameter code of daily flow.

        parameter_type = 'ST'  # The parameter type code which is only streamflow.

        parameter_mean = 3  # The code for mean of streamflow for each day.

        if state != '' and basin == '':  # Checks if user wants the stations based on basin or state.
            site_names = nwis.what_sites(stateCd=state, parameterCd=parameter_code, outputDataTypeCd='dv',
                                         startDT=start_date, endDT=end_date)[0]

        elif basin != '' and state == '':
            site_names = nwis.what_sites(huc=basin, parameterCd=parameter_code, outputDataTypeCd='dv',
                                         startDT=start_date, endDT=end_date)[0]

        site_names['parm_cd'] = site_names['parm_cd'].astype(np.int64)

        if status == 'all':  # Checks if user wants all the stations or only the ones suitable for SDF curve generation.
            site_names = site_names[(site_names['parm_cd'] == 60)].reset_index(drop=True)

        elif status == 'optimal':
            site_names = site_names[(site_names['parm_cd'] == 60) &
                                    (site_names['count_nu'] >= 10958)].reset_index(drop=True)

        site_names = site_names[(site_names['stat_cd'] == parameter_mean) &  # Modifies the list based on the parameter
                                (site_names['site_tp_cd'] == parameter_type)]  # features

        site_names[['begin_date', 'end_date']] = site_names[['begin_date', 'end_date']]. \
            apply(pd.to_datetime, errors='coerce')  # Changes the type of the begin_date column

        # Modifies the list based on the least number of years required for that specific state.
        site_names = site_names[(site_names['begin_date'].dt.year <= datetime.today().year -
                                 sufficient_year_number[0])].reset_index(drop=True)

        return site_names

    # The function for getting the streamflow data for a specific station.
    def daily_data(site='', start_date="1800-01-01", end_date=datetime.today().strftime('%Y-%m-%d')):
        parameter_code = "00060"

        stat_cd = "00003"

        daily_streamflow = nwis.get_dv(sites=site, parameterCd=parameter_code, start=start_date, end=end_date,
                                       statCd=stat_cd)[0]

        if daily_streamflow.shape[1] > 3:

            print('Corrupted Station Data Warning: The site does not have a compatible record for SDF '
                  'curve generation.\nPlease modify it manually or use another site number.')

        elif daily_streamflow.shape[1] == 0:

            print('Corrupted Station Data Warning: The site does not have a compatible record for SDF '
                  'curve generation.\nPlease modify it manually or use another site number.')

        else:

            daily_streamflow = daily_streamflow.reset_index()

            daily_streamflow.columns = ['Datetime', 'USGS_flow', 'qualifiers', 'USGS_ID']

            daily_streamflow['variable'] = 'streamflow'

            daily_streamflow.USGS_ID = daily_streamflow.USGS_ID.apply('="{}"'.format)

            daily_streamflow['measurement_unit'] = 'ft\u00b3/s'

            daily_streamflow = daily_streamflow[['Datetime', 'USGS_flow', 'variable', 'USGS_ID', 'measurement_unit',
                                                 'qualifiers']]

        return daily_streamflow

    #  This function creates the SDF curves for USGS gage data or any file.
    def sdf_creator(status='all', site='', data='', duration='all', figure=True):

        sufficient_year_number = [30]  # The least number of years which is sufficient for SDF curve creation.

        if len(site):  # Checks to see whether the user asks for USGS gage or a file related SDF curve.

            raw_data = pydat.daily_data(site=site)  # Calls the function to get the daily USGS gage data.

            if raw_data.shape[1] != 6 or raw_data.shape[1] == 0:  # Checks to whether data is ok or not.

                sys.exit(1)

        elif len(data):

            data = data.rename({'Flow':'USGS_flow'}, axis = 1)

            data['Datetime'] = pd.to_datetime(data['Datetime'])

            raw_data = data

        year_range = raw_data['Datetime'].dt.year.unique()  # Finds the years of the time-series.

        max_year = year_range[-1]  # The last year of the time-series.

        min_year = year_range[0]  # The first year of the time-series.

        max_month = raw_data.iloc[-1, 0].month  # The first month of the time-series.

        min_month = raw_data.iloc[0, 0].month  # The first month of the time-series.

        if max_month > 10:  # Checks to see whether the last year has the complete water year data.

            raw_data = raw_data.loc[(raw_data['Datetime'] < str(max_year) + '-10-01')]

        else:

            raw_data = raw_data.loc[(raw_data['Datetime'] < (str(year_range[-2]) + '-10-01'))]

            max_year = year_range[-2]  # If last year is not complete (does not have October), it goes one year back.

        if min_month < 10:  # Checks to see whether the first year has the complete water year data.

            raw_data = raw_data.loc[(raw_data['Datetime'] >= (str(min_year) + '-10-01'))]

        else:

            raw_data = raw_data.loc[(raw_data['Datetime'] >= (str(year_range[1]) + '-10-01'))]

            min_year = year_range[1]  # If first year is not complete (does not have October), it goes one year forward.

        if status == 'optimal':  # Checks whether the user wants the SDF curve based on the best data criteria. *

            # Whether data is more than the least number of required years.
            if len(year_range) < sufficient_year_number[0]:

                print('Lack of Sufficient Number of Years Warning: The number of years is less than sufficient.')

            # Whether it has the sufficient recent years.
            elif not any(x in year_range for x in list(range(datetime.today().year - sufficient_year_number[0] - 1,
                                                             datetime.today().year + 1))):
                print('Lack of Most Recent Data Warning: The data does not contain the last recent years.')

            else:
                min_year = year_range[-sufficient_year_number[0]]  # Determines the first of the updated time-series.

                # Uses the minimum number of years needed.
                raw_data = raw_data[(raw_data['Datetime'] >= (str(min_year) + '-10-01'))]

                year_range = raw_data['Datetime'].dt.year.unique()  # Updates the years list of time-series.

        mean_year = np.zeros((len(year_range) - 1, 5))

        for ii in range(len(mean_year)):  # Writes the year and the average value of it.

            mean_year[ii, 0] = int(year_range[ii + 1])  # Year number.

            mean_year[ii, 1] = (
                raw_data[(raw_data['Datetime'] >= (str(year_range[ii]) + '-10-01')) &  # Mean of that year.
                         (raw_data['Datetime'] < (str(year_range[ii + 1]) + '-10-01')) &
                         (raw_data['USGS_flow'] >= 0)]['USGS_flow'].mean())  # Neglects the negative data.

        mean_year = mean_year[~(np.isnan(mean_year)).any(axis=1)]  # Remove the NANs.

        overall_average = mean_year[:, 1].mean()  # The average of the whole data set.

        if duration == 'all':  # Checks which duration user asked for.

            duration = list(range(2, 11))

        else:

            duration = list(map(int, duration.split(',')))

        # Pre-defined colors for the SDF curve.
        color = ['b', 'g', 'y', 'r', 'orange', 'brown', 'gray', 'cyan', 'olive', 'pink']

        # Calculates the SDF curve for each duration.
        for dd in range(len(duration)):

            arrays = [['Duration=' + str(duration[dd])] * 5,
                      ["Date", "Mean_Flow(cfs)", 'Rolling_Average(cfs)', 'Severity(%)',
                       'Probability(%)']]

            tuples = list(zip(*arrays))

            index = pd.MultiIndex.from_tuples(tuples)

            mean_year_temp = pd.DataFrame(mean_year)

            # Calculates the moving average of required duration.
            mean_year_temp.iloc[:, 2] = mean_year_temp.iloc[:, 1].rolling(duration[dd]).mean()

            mean_year_temp = mean_year_temp.dropna()  # Remove the NANs of each moving average.

            # Calculates the severity.
            mean_year_temp.iloc[:, 3] = (mean_year_temp.iloc[:, 2] - overall_average) / overall_average * 100

            mean_year_temp = mean_year_temp[mean_year_temp.iloc[:, 3] <= 0]  # Removes the non drought severity.

            # Sort the data for frequency calculation
            temp_severity = mean_year_temp.sort_values(by=3, axis=0, ascending=False)

            for i in range(len(mean_year_temp)):  # Calculates the non-exceedance probability.

                temp_severity.iloc[i, 4] = (i + 1) / len(mean_year_temp) * 100

            temp_severity = temp_severity.sort_values(by=0, axis=0)

            temp_final = pd.DataFrame(temp_severity.to_numpy(), columns=index)

            if dd == 0:

                final = temp_final.reset_index(drop=True)

            else:

                final = pd.concat([final.reset_index(drop=True),
                                   temp_final.reset_index(drop=True)], axis=1)

        # Plotting the SDF curve.
        fig = 0

        # plt.rcParams["font.family"] = "Times New Roman"

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

        for dd, ii in enumerate(duration):
            filled_marker_style = dict(marker='o', linestyle='-', markersize=5,
                                       color=color[dd])

            temp_final = final[('Duration=' + str(ii))].sort_values(by=['Probability(%)'])

            axs.plot(temp_final.iloc[:, 3] * (-1), temp_final.iloc[:, 4], **filled_marker_style,
                     label=('Duration = ' + str(ii)))

        plt.legend(loc='lower right')

        if figure is True:

            plt.show()

        elif figure is False:

            plt.close()

        return final, raw_data, fig

    def info(tyd, limit):

        c = 1

        for i in range(len(tyd)):

            result_info_test = np.zeros([len(tyd), 3])

            b = 1

            index_1 = ['index_year']

            for j in range(len(tyd)):

                if j > i:

                    if tyd[i, 1] - limit <= tyd[j, 1] <= tyd[i, 1] + limit:
                        result_info_test[0, 0] = tyd[i, 0]

                        result_info_test[0, 1] = tyd[i, 1]

                        result_info_test[0, 2] = tyd[i, 2]

                        result_info_test[b, 0] = tyd[j, 0]

                        result_info_test[b, 1] = tyd[j, 1]

                        result_info_test[b, 2] = tyd[j, 2]

                        index_1 = index_1 + ["similar_year_" + str(b)]

                        b = b + 1

            result_info_test = np.delete(result_info_test, np.s_[b:len(tyd)], axis=0)

            if b != 1:

                arrays = [[str(int(result_info_test[0, 0]))] * b, index_1]

                tuples = list(zip(*arrays))

                index = pd.MultiIndex.from_tuples(tuples)

                df_1 = pd.DataFrame(result_info_test, index=index, columns=['year', 'severity', 'frequency'])

                if c == 1:
                    analog_year_info = pd.DataFrame(result_info_test, index=index,
                                                    columns=['year', 'severity', 'frequency'])

                if c > 1:
                    analog_year_info = pd.concat([analog_year_info, df_1], axis=0)

                c = c + 1

        return analog_year_info

    def matrix(tyd, limit):

        similar_year = np.zeros([len(tyd), len(tyd) - 1])    # Creating empty numpy array

        high_a = 0      # Creating an initial variable

        tyd[:, 1] = np.round(tyd[:, 1], 1)      # Rounding the severity to one decimal

        for i in range(len(tyd)):

            a = 1       # Creating an initial variable

            for j in range(len(tyd)):

                if j > i:       # This comparison is for preventing duplicate analog time-series

                    if tyd[i, 1] - limit <= tyd[j, 1] <= tyd[i, 1] + limit:     # Determining the analog time-series
                        # based on severity

                        similar_year[i, 0] = tyd[i, 0]      # The year of the SDF curve point

                        similar_year[i, a] = tyd[j, 0]      # The

                        a = a + 1

            # Find the highest number of analog time-series.

            if high_a < a:

                high_a = a

        similar_year[:, 0] = tyd[:, 0]

        similar_year = np.delete(similar_year, np.s_[high_a:len(tyd)], axis=1)

        analog_year_matrix = pd.DataFrame(similar_year, columns=['index year'] + ['similar year'] * (high_a - 1))

        return analog_year_matrix

    def analog(tyd, limit, duration, ts):

        c = 1

        for i in range(len(tyd)):

            b = 1

            for j in range(len(tyd)):

                if j > i:

                    if tyd[i, 1] - limit <= tyd[j, 1] <= tyd[i, 1] + limit:

                        if b == 1:

                            end_year = int(tyd[i, 0])

                            start_year = end_year - duration

                            index_year = ts[(ts.iloc[:, 0] >= str(start_year) + '-10-01') & (ts.iloc[:, 0] < str(end_year) +
                                                                                             '-10-01')]

                            arrays = [[str(int(tyd[i, 0]))] * 2, ["Date_index", "Flow_cfs_index"]]

                            tuples = list(zip(*arrays))

                            index = pd.MultiIndex.from_tuples(tuples)

                            index_year = pd.DataFrame(index_year.to_numpy(), columns=index)

                        # Analog time-series

                        end_year = int(tyd[j, 0])

                        start_year = end_year - duration

                        result_year = ts[(ts.iloc[:, 0] >= str(start_year) + '-10-01') & (ts.iloc[:, 0] < str(end_year) +
                                                                                          '-10-01')]

                        arrays = [[str(int(tyd[i, 0]))] * 2, ["Date_similar_" + str(b), "Flow_cfs_similar_" + str(b)]]

                        tuples = list(zip(*arrays))

                        index = pd.MultiIndex.from_tuples(tuples)

                        result_year = pd.DataFrame(result_year.to_numpy(), columns=index)

                        index_year = pd.concat([index_year.reset_index(drop=True), result_year.reset_index(drop=True)],
                                               axis=1)

                        b = b + 1

            if b != 1:

                if c == 1:

                    analog_year_series = index_year

                if c > 1:

                    analog_year_series = pd.concat([analog_year_series.reset_index(drop=True),
                                                    index_year.reset_index(drop=True)], axis=1)

                c = c + 1

        return analog_year_series

    # This function generates the streamflow analogs.
    def streamflow_analog(status='all', site='', data='', duration=''):
        # Call the sdf_creator function to get the raw data and sdf curve data.
        sdf_data, raw_data, fig = pydat.sdf_creator(status=status, site=site, data=data, duration=duration, figure=False)

        duration = duration  # Asking the duration of the SDF curve.

        limit = 0.5  # Asking the limitation value for severity index comparison.

        # Getting the sdf curve data related to the required duration
        modified_sdf = sdf_data[('Duration=' + str(duration))]

        #  Getting the proper columns and turning them to numpy for higher calculation speed.
        tyd = modified_sdf.loc[:, ('Date', 'Severity(%)', 'Probability(%)')].to_numpy()

        ts = raw_data.iloc[:, 0:2]  # Separating the date and streamflow data from the whole data.

        analog_year = \
            pydat.matrix(tyd, limit)  # Function for creating a table indicating the points with analog time-series.

        # Function for finding the information of each point and its analog time-series
        analog_year_info = pydat.info(tyd, limit)

        # Function for finding the analog time-series daily data
        analog_year_series = pydat.analog(tyd, limit, int(duration), ts)

        #  Returning the analog years' list, analog years' information, and the daily data for analog years.
        return analog_year, analog_year_info, analog_year_series
