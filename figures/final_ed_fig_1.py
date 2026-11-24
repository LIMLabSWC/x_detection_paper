# =========================================================================
# Standalone Code to Reproduce Panels a and d–h for Fig 1 Script
# =========================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as tck
from analysis_scripts.plot_funcs import format_axis
from analysis_scripts.plot_time_resolved_decoding import unique_mice

all_sess_td_df = pd.read_hdf('all_td_df_4_beh.h5')
assert isinstance(all_sess_td_df, pd.DataFrame)

# Ensure good trial filter is defined
good_trial_filt = 'Stage>=3 & n_since_last_Trial_Outcome <=5'

plt.style.use('figure_stylesheet.mplstyle')

boxplot_kwargs = dict(
    widths=0.5,
    patch_artist=True,
    showmeans=False,
    showfliers=False,
    medianprops=dict(lw=1, color='red'),
    boxprops=dict(lw=0.5, facecolor='white'),
    whiskerprops=dict(lw=0.5),
    capprops=dict(lw=0.5),
)

# -------------------------------------------------------------------------
# Panel a: Example Training Progression (Hit rate & early lick rate across sessions)
# -------------------------------------------------------------------------

all_pattern_trials_df = all_sess_td_df.query(
        'Stage>=3 & Tone_Position==0 & n_since_last_Trial_Outcome<=5')

unique_mice = all_sess_td_df.query('Stage==2').index.get_level_values('name').unique()
for eg_mouse in unique_mice[:1]:
    fig_a, ax_a = plt.subplots(figsize=(2.0, 1.5))
    # eg_mouse = unique_mice[]
    eg_mouse_perf = all_sess_td_df.query('10<trial_num<=250 & Stage>=2').xs(eg_mouse, level='name').groupby(level=['sess'])[
        'Trial_Outcome'].mean()
    eg_mouse_early_licks = all_sess_td_df.query('10<trial_num<=250 & Stage>=2').xs(eg_mouse, level='name').groupby(level=['sess'])[
        'Early_Lick_bool'].mean()
    eg_mouse_stage = \
    all_sess_td_df.query('n_since_last_Trial_Outcome <=5 & Stage>=2').xs(eg_mouse, level='name').groupby(level=['sess'])[
        'Stage'].mean()

    ax_a.plot(np.arange(len(eg_mouse_perf)), eg_mouse_perf, c='k', lw=1.5)
    for d, stage in enumerate(eg_mouse_stage):
        ax_a.axvline(d, c=f'C{int(stage)}', lw=4, alpha=0.3)

    ax2_a = ax_a.twinx()
    ax2_a.plot(np.arange(len(eg_mouse_perf)), eg_mouse_early_licks, c='k', lw=1.2, ls='--')
    ax2_a.set_ylim(0, 1)
    ax2_a.spines['right'].set_visible(True)

    ax_a.set_xlim(-1, 10)
    ax_a.set_ylim(0.3, 1.01)
    ax_a.set_ylabel('Hit rate')
    ax_a.set_xlabel('Session Number')
    fig_a.set_layout_engine('tight')
    fig_a.show()

# -------------------------------------------------------------------------
# Panel c: Hazard Rate
# -------------------------------------------------------------------------
hazard_rate_td_df = pd.read_hdf('hazard_rate_td_df.h5')
assert isinstance(hazard_rate_td_df, pd.DataFrame)

# 1. Total initiated trials (at-risk pool at t=0)
trial_level_df = hazard_rate_td_df.groupby(level=hazard_rate_td_df.index.names).first()
n_total_trials = len(trial_level_df)

# 2. Extract event times relative to trial start
patt_trials = hazard_rate_td_df.loc[hazard_rate_td_df['Tone_Position'] == 0]

patt_from_start_dt = (
    patt_trials['ToneTime_dt'] - patt_trials['Trial_Start_dt']
).dt.total_seconds()
x_from_start_dt = (
    trial_level_df['Gap_Time_dt'] - trial_level_df['Trial_Start_dt']
).dt.total_seconds()


# Helper function to compute hazard rate
def compute_hazard_rate(
    event_times: pd.Series,
    n_total: int,
    bin_width: float = 0.05,
    max_t: float = None,
):
    times = event_times.dropna().values
    times = times[times > 0]
    if max_t is None:
        max_t = np.quantile(times, 0.98)

    times = times[times <= max_t]
    bins = np.arange(0, max_t + bin_width, bin_width)

    counts, bin_edges = np.histogram(times, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    cum_events = np.cumsum(counts)
    n_at_risk = n_total - np.concatenate(([0], cum_events[:-1]))

    hazard = np.divide(
        counts,
        n_at_risk * bin_width,
        out=np.zeros_like(counts, dtype=float),
        where=n_at_risk > 0,
    )

    return bin_centers, hazard


# 3. Calculate hazard rates across a shared max time limit
bin_width = 1  # 50 ms
max_t = patt_from_start_dt.max()

t_patt, h_patt = compute_hazard_rate(
    patt_from_start_dt, n_total_trials, bin_width=bin_width, max_t=max_t
)
t_x, h_x = compute_hazard_rate(
    x_from_start_dt, n_total_trials, bin_width=bin_width, max_t=x_from_start_dt.quantile(0.99)
)

# 4. Plot using Figure and Axes object methods exclusively
fig_c, ax_c = plt.subplots()
fig_c.set_size_inches(1.7, 2.5)
fig_c.set_layout_engine('tight')

ax_c.plot(t_patt, h_patt, color='k', lw=2, label='Pattern')
ax_c.plot(t_x, h_x, color='purple', lw=2, linestyle='-', label='X')

ax_c.set_ylim(bottom=0)
ax_c.legend(frameon=False)
format_axis(ax_c, ylabel='Hazard rate ($s^{-1}$)', xlabel='Time from trial start (s)')
ax_c.set_xscale('symlog', base=2, linthresh=5)

fig_c.show()

# -------------------------------------------------------------------------
# Panel d: Reaction Time by Number of Tones Played (AB, ABC, ABCD)
# -------------------------------------------------------------------------
all_correct_trials_df = all_sess_td_df.query('Stage>=3 & Trial_Outcome==1').copy()
all_correct_trials_df['reaction_time'] = (
    all_correct_trials_df['Trial_End_dt'] - all_correct_trials_df['Gap_Time_dt']
).dt.total_seconds()

fig_d, ax_d = plt.subplots(figsize=(1.5, 1.5))

means_by_group_tones = all_correct_trials_df.groupby(['name', 'N_TonesPlayed'])['reaction_time'].mean().unstack()[
    [2, 3, 4]].dropna()
bxplot_d = ax_d.boxplot(means_by_group_tones, labels=['AB', 'ABC', 'ABCD'], **boxplot_kwargs)
ax_d.set_ylabel('Reaction Time (s)')
ax_d.set_ylim(0, 0.5)
ax_d.locator_params(axis='y', nbins=4)
format_axis(ax_d)
fig_d.set_layout_engine('tight')
fig_d.show()

# -------------------------------------------------------------------------
# Panel e: Reaction Time by Tone Position (Pattern vs Non-pattern)
# -------------------------------------------------------------------------
fig_e, ax_e = plt.subplots(figsize=(1.25, 1.5))
means_by_group_pos = all_correct_trials_df.groupby(['name', 'Tone_Position'])['reaction_time'].mean().unstack()[
    [1, 0]].dropna()
bxplot_e = ax_e.boxplot(means_by_group_pos, labels=['Pattern', 'Non pattern'], **boxplot_kwargs)
ax_e.set_ylabel('Reaction Time (s)')
ax_e.set_ylim(0, 0.5)
ax_e.locator_params(axis='y', nbins=4)
format_axis(ax_e)
fig_e.set_layout_engine('tight')
fig_e.show()

# -------------------------------------------------------------------------
# Panels f, g, h: Performance vs Time intervals (Trial Start to X, Pattern to X, Trial Start to Pattern)
# -------------------------------------------------------------------------
time_to_metrics = {
    'start_to_gap': ('Trial start to X onset (s)', 'start_to_X_onset.pdf'),
    'time_to_gap': ('Pattern onset to X onset (s)', 'pattern_to_X_onset.pdf'),
    'time_to_tone': ('Trial start to pattern onset (s)', 'trial_start_to_pattern.pdf')
}

for metric, (x_label, fname) in time_to_metrics.items():
    if metric in all_pattern_trials_df.columns:
        fig_t, ax_t = plt.subplots(figsize=(1.8, 1.5))
        pattern_trials_by_time_to = all_pattern_trials_df.groupby(['name', f'{metric}_rounded'])[
            'Trial_Outcome'].mean().unstack()
        times2use = all_pattern_trials_df[f'{metric}_rounded'].value_counts().where(
            lambda x: x > 0.03 * len(all_pattern_trials_df)).dropna().index.sort_values().values

        filtered_data = [pattern_trials_by_time_to[t].dropna().values for t in times2use if
                         t in pattern_trials_by_time_to]
        if filtered_data:
            bxplot_t = ax_t.boxplot(filtered_data, labels=times2use.astype(int), **boxplot_kwargs)
            ax_t.set_ylim(0.5, 1.01)
            ax_t.set_ylabel('Performance')
            ax_t.set_xlabel(x_label)
            ax_t.locator_params(axis='y', nbins=4)
            format_axis(ax_t)
            fig_t.set_layout_engine('tight')
            fig_t.show()

