# %% draw the bar chart


for state_name in states:

best_bic = best_bic[best_bic > 2]
best_aic = best_aic[best_aic > 2]
best_p_value = best_p_value[best_p_value > 2]

best_bic = best_bic.reset_index(name='values').sort_values(by=0)
best_aic = best_aic.reset_index(name='values').sort_values(by=0)
best_p_value = best_p_value.reset_index(name='values').sort_values(by=0)

df = pd.DataFrame({"group": ["aic"] * len(best_aic) + ["bic"] * len(best_bic) + ["ks test"] * len(best_p_value)})
VALUES = []
LABELS = []
for i in [best_aic, best_bic, best_p_value]:
    for j in range(len(i)):
        VALUES.append(i.iloc[j, 1])

for i in [best_aic, best_bic,
          best_p_value]:
    for j in range(len(i)):
        LABELS.append(i.iloc[j, 0])

# Determine the width of each bar.
# The circumference is '2 * pi', so we divide that total width over the number of bars.
WIDTH = 2 * np.pi / len(VALUES)

# Determines where to place the first bar.
# By default, matplotlib starts at 0 (the first bar is horizontal)
# but here we say we want to start at pi/2 (90 deg)
OFFSET = np.pi / 2


def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else:
        alignment = "left"
    return rotation, alignment


def add_labels(angles, values, labels, offset, ax):
    # This is the space between the end of the bar and the label
    padding = 4

    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle

        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle,
            y=value + padding,
            s=label,
            ha=alignment,
            va="center",
            rotation=rotation,
            rotation_mode="anchor"
        )


# Grab the group values
GROUP = df["group"].values

# Add three empty bars to the end of each group
PAD = 3
ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
WIDTH = (2 * np.pi) / len(ANGLES)

# Obtain size of each group
GROUPS_SIZE = [len(i[1]) for i in df.groupby("group")]

# Obtaining the right indexes is now a little more complicated
offset = 0
IDXS = []
for size in GROUPS_SIZE:
    IDXS += list(range(offset + PAD, offset + size + PAD))
    offset += size + PAD

# Same layout as above
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.set_theta_offset(OFFSET)
ax.set_ylim(-100, 100)
ax.set_frame_on(False)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_yticks([0, 10, 20, 100])
# Use different colors for each group!
GROUPS_SIZE = [len(i[1]) for i in df.groupby("group")]
COLORS = [f"C{i}" for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]

# And finally add the bars.
# Note again the `ANGLES[IDXS]` to drop some angles that leave the space between bars.
ax.bar(
    ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS,
    edgecolor="white", linewidth=2
)

add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)

# Extra
offset = 0
for group, size in zip(["aic", "bic", "ks"], GROUPS_SIZE):
    # Add line below bars
    x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
    ax.plot(x1, [-5] * 50, color="#333333")

    # Add text to indicate group
    ax.text(
        np.mean(x1), -20, group, color="#333333", fontsize=14,
        fontweight="bold", ha="center", va="center"
    )

    # Add reference lines at 20, 40, 60, and 80
    x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
    ax.plot(x2, [0] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [30] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [60] * 50, color="#bebebe", lw=0.8)

    offset += size + PAD

    if group == 'aic':
        ax.text(np.mean(x2), 0 + PAD, "0", ha="center", size=8)
        ax.text(np.mean(x2), 30 + PAD, "30", ha="center", size=8)
        ax.text(np.mean(x2), 60 + PAD, "60", ha="center", size=8)

    ax.set(
        title='Goodness of Fit Tests for ' + str(len(valid_statin)) + ' Stations in Alabama State (10 Years Duration)')

write_path = newpath + 'distribution_search_result'

plt.savefig('al_10_fit_distribution.png')
'''

# %%

'''  # %%
valid_statin = valid_station(0, 60, states)

for state_name in states:

    # set the directory name
    parent_dir = onedrive_path + state_name

    state_data[state_name] = {}

    # read each file and get data and generate the sdf curve
    for jj in range(len(valid_statin)):
        # get the station name from the file names

        # files = os.get_files_in_directory(parent_dir + '/')

        csv_files = glob.glob(os.path.join(parent_dir, "*.csv"))

        csv_files.remove(parent_dir + '\station information.csv')

        temp_index = [i for i, s in enumerate(csv_files) if valid_statin.iloc[jj, 0] in s]

        raw_df = pd.read_csv(csv_files[temp_index[0]], encoding='unicode_escape')

        station_name = valid_statin.iloc[jj, 0]

        state_data[state_name][station_name] = pydat.sdf_creator(data=raw_df, figure=False)[0]

    for station in valid_statin['station']:

        print(station)

        for duration in range(2, 3):
            data = state_data[state_name][station]['Duration=' + str(duration)]

            params = stats.kappa3.fit(data.iloc[:, 3].dropna() * -1)
            # Get the cumulative distribution function (CDF) for the fitted distribution
            cdf = stats.kappa3.cdf(data.iloc[:, 3].dropna() * -1, *params)

            fig, axs = plt.subplots(dpi=300)
            stats.probplot(data.iloc[:, 3].dropna() * -1, sparams=params, dist=stats.kappa3.name, plot=axs)
            axs.set(title=station)
            plt.savefig(
                '/Users/savalan/Library/CloudStorage/OneDrive-TheUniversityofAlabama/02.projects/kappa3/' + station + ' - kappa3')
            '''

# %%
'''
            if ks_statistic < best_ks_statistic:
    best_fit = distribution
    best_fit_params = params
    best_ks_statistic = ks_statistic
    '''


    #%% test
'''
station_best_result = []
station_best_result_value = []

for state_name in states:
    for zz, station in enumerate(valid_statin['station']):
        for duration in range(2, 3):
            distribution_data_temp = distribution_data[state_name][station][duration].reset_index(drop=True)

            if (distribution_data_temp.iloc[:, 2] < 0.05).any():
                print(station, 'yes')

            station_best_result.append(distribution_data_temp.iloc[:3, 0])

            station_best_result_value.append(distribution_data_temp.iloc[:3, 2])

aa = pd.DataFrame(station_best_result)

aa_1 = pd.DataFrame(station_best_result_value)

bb = aa.iloc[:, 0].value_counts()

bb_1 = aa_1.iloc[:, 0].value_counts()

cc = aa.iloc[:, 1].value_counts()

dd = aa.iloc[:, 2].value_counts()
'''

'''
distributions_all = [
    stats.norm,  # Normal
    stats.lognorm,  # Log-Normal
    stats.pearson3,  # Pearson III
    stats.gennorm,  # Generalized Normal
    stats.genextreme,  # Generalized Extreme
    stats.genlogistic,  # Generalized Logistic
    stats.genpareto,  # Generalized Pareto
    stats.kappa3,  # Kappa 3
    stats.expon,  # Exponential
    stats.gamma,  # Gamma
    stats.beta,  # Beta
    stats.weibull_min,  # Weibull Minimum
    stats.weibull_max,  # Weibull Maximum
    stats.pareto,  # Pareto
    stats.uniform  # Uniform
    # Add more distributions here if needed
]
'''