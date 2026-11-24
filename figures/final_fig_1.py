import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as tck

import numpy as np
import pandas as pd

from analysis_scripts.pupil_analysis_funcs import PupilCondAnalysis
from xdetectioncore.io_utils import load_response_dfs
from xdetectioncore.behaviour import get_all_cond_filts
from xdetectioncore.plotting import format_axis, plot_shaded_error_ts

if __name__ == '__main__':

    plt.style.use('../figure_stylesheet.mplstyle')

    # Load pre-computed condition response dictionaries
    A_by_cond_path = Path(r"X:\Xdetection_data\data_for_paper\fig1_A_by_cond.h5")
    X_by_cond_path = Path(r"X:\Xdetection_data\data_for_paper\fig1_X_by_cond.h5")

    A_by_cond = load_response_dfs(A_by_cond_path)
    X_by_cond = load_response_dfs(X_by_cond_path)

    # Load trial data
    all_td_df_path = Path(r"X:\Xdetection_data\data_for_paper\all_td_df.h5")
    all_td_df = pd.read_hdf(all_td_df_path)
    assert isinstance(all_td_df, pd.DataFrame)

    cond_filters = get_all_cond_filts()
    good_trial_filt = 'n_since_last_Trial_Outcome <=5 '

    # Load session lists by condition
    sesslist_path = Path('sesslist_by_cond.json')
    if sesslist_path.is_file():
        with open(sesslist_path, 'r') as f:
            sesslist_by_cond = json.load(f)
        all_rare_freq_sess = sesslist_by_cond['event_rate']
        norm_dev_sesslist = sesslist_by_cond['norm_dev']
        dev_sesslist = sesslist_by_cond['generalisation']
    else:
        # Fallback to unique sessions found across keys
        all_rare_freq_sess = list(A_by_cond['rare'].index.get_level_values('sess').unique())
        norm_dev_sesslist = list(A_by_cond['normal'].index.get_level_values('sess').unique())
        dev_sesslist = list(A_by_cond['dev_ABCD1'].index.get_level_values('sess').unique())

    # Common styling kwargs
    boxplot_kwargs = dict(
        widths=0.5,
        patch_artist=False,
        showmeans=False,
        showfliers=False,
        medianprops=dict(mfc='k'),
        boxprops=dict(lw=0.5),
        whiskerprops=dict(lw=0.5),
        capprops=dict(lw=0.5),
    )

    xlim = [-0.25, 2.25]

    # =========================================================================
    # 1. X Response Plot (hit_all vs miss_all)
    # =========================================================================
    x_sess_list = X_by_cond['miss_all'].index.get_level_values('sess').unique()
    X_resp = PupilCondAnalysis(
        by_cond_dict=X_by_cond,
        sess_list=x_sess_list,
        conditions=['hit_all', 'miss_all'],
        event_name='X'
    )
    x_line_kwargs = {
        'hit_all': {'c': '#31572c', 'ls': '-'},
        'miss_all': {'c': '#31572c', 'ls': '--'}
    }
    X_resp.plot_ts_by_cond(cond_line_kwargs=x_line_kwargs)
    X_resp.plots['ts_by_cond'][0].set_size_inches(1.5, 2)
    X_resp.plots['ts_by_cond'][0].show()
    X_resp.plots['ts_by_cond'][0].savefig('X_resp.pdf')

    # =========================================================================
    # 2. X Intermediate Responses Plot (AB, ABC, ABCD)
    # =========================================================================
    if all(cond in X_by_cond for cond in ['AB', 'ABC', 'ABCD']):
        x_inter_resps = PupilCondAnalysis(
            by_cond_dict=X_by_cond,
            conditions=['AB', 'ABC', 'ABCD'],
            event_name='X'
        )
        early_x_line_kwargs = {
            'AB': {'c': '#31572c', 'ls': '-'},
            'ABC': {'c': '#84a98c', 'ls': '-'},
            'ABCD': {'c': '#c2c5aa', 'ls': '-'}
        }
        x_inter_resps.plot_ts_by_cond(cond_line_kwargs=early_x_line_kwargs)
        x_inter_resps.plots['ts_by_cond'][0].set_size_inches(2, 2)
        x_inter_resps.plots['ts_by_cond'][0].show()
        x_inter_resps.plots['ts_by_cond'][0].savefig('x_inter_resp.pdf')

    # =========================================================================
    # 3. Rare vs Frequent Analysis
    # =========================================================================

    rare_freq_line_kwargs = {
        'rare': {'c': '#00cbfcff', 'ls': '-'},
        'frequent': {'c': '#1530e9ff', 'ls': '-'}
    }
    rare_freq_condnames = ['frequent', 'rare']

    fig_1e_plot_data = {}
    for stim_name, stim_by_cond in zip(['A', 'X'], [A_by_cond, X_by_cond]):
        rare_freq_analysis = PupilCondAnalysis(stim_by_cond, rare_freq_condnames,all_rare_freq_sess)
        rare_freq_analysis.plot_ts_by_cond(cond_line_kwargs=rare_freq_line_kwargs)
        rare_freq_analysis.plots['ts_by_cond'][1].set_xlim(xlim)
        rare_freq_analysis.plots['ts_by_cond'][0].set_size_inches(2, 2)
        rare_freq_analysis.plots['ts_by_cond'][0].show()

        for cond, cond_resps in rare_freq_analysis.by_cond.items():
            fig_1e_plot_data[f'{stim_name}_{cond}_mean'] = cond_resps.groupby('sess').mean().mean()
            fig_1e_plot_data[f'{stim_name}_{cond}_sem'] = cond_resps.groupby('sess').mean().sem()

    pd.DataFrame(fig_1e_plot_data).T.to_csv('fig_1e_plot_data.csv')

    # Behavior: Rare vs Frequent
    cond_td_df = all_td_df.loc[all_td_df.index.get_level_values('sess').isin(all_rare_freq_sess)]
    rare_freq_perf = [
        cond_td_df.query(good_trial_filt).query(cond_filters[cond])['Trial_Outcome'].groupby('name').mean()
        for cond in rare_freq_condnames
    ]
    fig_i, ax_i = plt.subplots()
    ax_i.bar(rare_freq_condnames, [e.mean() for e in rare_freq_perf], ec='black', fc='w', lw=0.5, width=0.4)
    for cond, data in zip(rare_freq_condnames, rare_freq_perf):
        ax_i.scatter([cond] * len(data), data, edgecolor='none', fc=rare_freq_line_kwargs[cond]['c'], alpha=0.5)

    format_axis(ax_i)
    fig_i.set_size_inches(2, 2)
    fig_i.set_layout_engine('constrained')
    fig_i.show()

    pd.DataFrame(rare_freq_perf).to_csv('fig_1i_rare_freq_perf.csv')

    # =========================================================================
    # 4. Normal vs Deviant Analysis
    # =========================================================================

    norm_dev_line_kwargs = {
        'normal': {'c': 'black', 'ls': '-'},
        'deviant_C': {'c': '#cd2727ff', 'ls': '-'}
    }
    norm_dev_cond_names = ['normal', 'deviant_C']

    fig_1f_plot_data = {}
    for stim_name, stim_by_cond in zip(['A', 'X'], [A_by_cond, X_by_cond]):
        norm_dev_analysis = PupilCondAnalysis(stim_by_cond, norm_dev_cond_names,norm_dev_sesslist)
        norm_dev_analysis.plot_ts_by_cond(cond_line_kwargs=norm_dev_line_kwargs)
        norm_dev_analysis.plots['ts_by_cond'][1].set_xlim(xlim)
        norm_dev_analysis.plots['ts_by_cond'][0].set_size_inches(2, 2)
        norm_dev_analysis.plots['ts_by_cond'][0].show()

        for cond, cond_resps in norm_dev_analysis.by_cond.items():
            fig_1f_plot_data[f'{stim_name}_{cond}_mean'] = cond_resps.groupby('sess').mean().mean()
            fig_1f_plot_data[f'{stim_name}_{cond}_sem'] = cond_resps.groupby('sess').mean().sem()

    pd.DataFrame(fig_1f_plot_data).T.to_csv('fig_1f_plot_data.csv')

    # Behavior: Normal vs Deviant
    cond_td_df = all_td_df.loc[all_td_df.index.get_level_values('sess').isin(norm_dev_sesslist)]
    norm_dev_perf = [
        cond_td_df.query(good_trial_filt).query(cond_filters[cond])['Trial_Outcome'].groupby('name').mean()
        for cond in norm_dev_cond_names
    ]

    fig_1k, ax_k = plt.subplots()
    ax_k.bar(norm_dev_cond_names, [e.mean() for e in norm_dev_perf], ec='black', fc='w', lw=0.5, width=0.4)
    for cond, data in zip(norm_dev_cond_names, norm_dev_perf):
        ax_k.scatter([cond] * len(data), data, edgecolor='none', fc=norm_dev_line_kwargs[cond]['c'], alpha=0.5)

    format_axis(ax_k)
    fig_1k.set_size_inches(1.5, 2)
    fig_1k.set_layout_engine('constrained')
    fig_1k.show()

    pd.DataFrame(norm_dev_perf).to_csv('fig_1k_norm_dev_perf.csv')

    # =========================================================================
    # 5. dev_ABCD1 and dev_ABBA1 Analysis
    # =========================================================================

    dev_cond_names = ['normal', 'dev_ABCD1', 'dev_ABBA1']
    dev_line_kwargs = {
        'normal': {'c': 'k', 'ls': '-'},
        'dev_ABCD1': {'c': '#e8739bff', 'ls': '-'},
        'dev_ABBA1': {'c': '#73e8c1ff', 'ls': '-'}
    }

    fig_1g_plot_data = {}
    dev_ABBA1_ABCD_analysis = PupilCondAnalysis(A_by_cond, dev_cond_names,dev_sesslist)
    dev_ABBA1_ABCD_analysis.plot_ts_by_cond(cond_line_kwargs=dev_line_kwargs)
    dev_ABBA1_ABCD_analysis.plots['ts_by_cond'][1].set_xlim(xlim)
    dev_ABBA1_ABCD_analysis.plots['ts_by_cond'][0].show()

    for cond, cond_resps in dev_ABBA1_ABCD_analysis.by_cond.items():
        fig_1g_plot_data[f'A_{cond}_mean'] = cond_resps.groupby('sess').mean().mean()
        fig_1g_plot_data[f'A_{cond}_sem'] = cond_resps.groupby('sess').mean().sem() 

    pd.DataFrame(fig_1g_plot_data).T.to_csv('fig_1g_plot_data.csv')

    # =========================================================================
    # Standalone Code to Reproduce Panels C and D for Fig 1 Script
    # =========================================================================


    all_sess_td_df = pd.read_hdf('all_td_df_4_beh.h5')

    # Ensure good trial filter is defined
    good_trial_filt = 'Stage>=3 & n_since_last_Trial_Outcome <=5'

    # -------------------------------------------------------------------------
    # Panel C: Hit rate comparison (Pattern vs Non-pattern) with paired lines
    # -------------------------------------------------------------------------
    fig_c, ax_c = plt.subplots(figsize=(1.25, 1.5))

    # Extract hit rate data per mouse/session for Pattern (Tone_Position == 1) and Non-pattern (Tone_Position == 0)
    perf_data = all_sess_td_df.query('Stage>=3& 20<trial_num<=250').groupby(['name', 'Tone_Position'])[
            'Trial_Outcome'].mean().unstack().dropna(axis=0)[[1, 0]].values

    # Plot summary bars
    ax_c.bar(['Pattern', 'Non-pattern'], perf_data.mean(axis=0),
            ec='black', fc='w', lw=0.5, width=0.4)

    # Scatter individual points and connect with paired lines across conditions
    patt_nonpatt_cols = ['dimgray', 'indigo']
    for cond_idx, (cond_name, cond_data, c) in enumerate(zip(['Pattern', 'Non-pattern'], perf_data.T, patt_nonpatt_cols)):
        ax_c.scatter([cond_idx] * len(cond_data), cond_data, facecolor=c, alpha=0.5, s=15, lw=0.01)

    for a_data in perf_data:
        ax_c.plot([0, 1], a_data, lw=0.25, color='gray')

    ax_c.set_ylim(0.4, 1.01)
    format_axis(ax_c)
    ax_c.set_ylabel('Hit rate')
    fig_c.set_layout_engine('tight')
    fig_c.show()

    pd.DataFrame(perf_data, columns=['Pattern', 'Non-pattern']).to_csv('fig_1c_pattern_nonpattern_perf.csv', index=True)

    # -------------------------------------------------------------------------
    # Panel D: Lick rate time-series (Top: licks to X onset, Bottom: licks to pattern onset)
    # -------------------------------------------------------------------------

    patt_non_patt_licks_df = pd.read_hdf('patt_non_patt_licks_df.h5','df')
    patt_non_patt2X_licks_df = pd.read_hdf('patt_non_patt2X_licks_df.h5','df')


    x_ser = np.array(patt_non_patt_licks_df.columns.total_seconds()).astype(float)
    y_lim = 5
    x_lim = (-0.5, 1.5)

    # --- Top Subplot: Licks to X onset (Pattern vs Non-pattern) ---
    fig_d_top, ax_d_top = plt.subplots(figsize=(1.25, 0.6))
    mean_by_trial_type_x = patt_non_patt2X_licks_df.groupby(['trial_type', 'sess']).mean()
    sem_by_trial_type_x = patt_non_patt2X_licks_df.groupby(['trial_type', 'sess']).sem()

    fig_1d_plot_data = {}

    for type_i, (trial_type, lbl) in enumerate(zip(['non_patt', 'patt'], ['non pattern', 'pattern'])):
        mean_ts = mean_by_trial_type_x.loc[trial_type].mean(axis=0)
        sem_ts = sem_by_trial_type_x.loc[trial_type].mean(axis=0)

        ax_d_top.plot(x_ser, mean_ts, label=lbl, c=patt_nonpatt_cols[type_i], lw=1)
        plot_shaded_error_ts(ax_d_top, x_ser, mean_ts.values, sem_ts.values, color=patt_nonpatt_cols[type_i], alpha=0.1)

        fig_1d_plot_data[f'X_{trial_type}_licks_mean'] = mean_ts.values
        fig_1d_plot_data[f'X_{trial_type}_licks_sem'] = sem_ts.values

    format_axis(ax_d_top)
    ax_d_top.set_ylabel('')
    ax_d_top.set_ylim(0, y_lim)
    ax_d_top.set_xlim(*x_lim)
    ax_d_top.xaxis.set_major_locator(tck.MultipleLocator())
    ax_d_top.set_xlabel('')
    ax_d_top.axvline(0, ls='--', c='k', lw=0.75)
    fig_d_top.set_layout_engine('constrained')
    fig_d_top.show()

    # --- Bottom Subplot: Licks to Pattern onset (Pattern vs Non-pattern) ---
    fig_d_bot, ax_d_bot = plt.subplots(figsize=(1.25, 0.6))
    mean_by_trial_type_p = patt_non_patt_licks_df.groupby(['trial_type', 'sess']).mean()
    sem_by_trial_type_p = patt_non_patt_licks_df.groupby(['trial_type', 'sess']).sem()

    for type_i, (trial_type, lbl) in enumerate(zip(['non_patt', 'patt'], ['non pattern', 'pattern'])):
        mean_ts = mean_by_trial_type_p.loc[trial_type].mean(axis=0)
        sem_ts = sem_by_trial_type_p.loc[trial_type].mean(axis=0)

        ax_d_bot.plot(x_ser, mean_ts, label=lbl, c=patt_nonpatt_cols[type_i], lw=1)
        plot_shaded_error_ts(ax_d_bot, x_ser, mean_ts.values, sem_ts.values, color=patt_nonpatt_cols[type_i], alpha=0.1)

        fig_1d_plot_data[f'pattern_{trial_type}_licks_mean'] = mean_ts.values
        fig_1d_plot_data[f'pattern_{trial_type}_licks_sem'] = sem_ts.values

    format_axis(ax_d_bot)
    ax_d_bot.set_ylabel('')
    ax_d_bot.set_ylim(0, y_lim)
    ax_d_bot.set_xlim(*x_lim)
    ax_d_bot.xaxis.set_major_locator(tck.MultipleLocator())
    ax_d_bot.set_xlabel('')
    ax_d_bot.axvline(0, ls='--', c='k', lw=0.75)
    fig_d_bot.set_layout_engine('constrained')
    fig_d_bot.show()

    pd.DataFrame(fig_1d_plot_data).T.to_csv('fig_1d_licks_ts_plot_data.csv', index=True)
    # =========================================================================
    # Panels L & M: Pre-pattern and Pre-X Pupil Size Boxplots by Expectancy
    # =========================================================================

    # Rename and use A_by_cond / X_by_cond as starting point
    A_by_cond_pre = load_response_dfs(Path('arousal_control_A_by_cond.h5'))
    X_by_cond_pre = load_response_dfs(Path('arousal_control_X_by_cond.h5'))

    conds_by_expected = {
        'expected': ['normal', 'frequent'],
        'unexpected': ['deviant_C', 'rare']
    }

    # Tag and concatenate expected vs unexpected conditions for Pattern (A)
    for exp_type, cond_list in conds_by_expected.items():
        exp_val = 1 if exp_type == 'expected' else 0
        for c in cond_list:
            if c in A_by_cond_pre:
                A_by_cond_pre[c]['expected'] = exp_val
            if c in X_by_cond_pre:
                X_by_cond_pre[c]['expected'] = exp_val

    A_resps_exp = pd.concat([A_by_cond_pre[c] for c in conds_by_expected['expected'] if c in A_by_cond_pre])
    A_resps_unexp = pd.concat([A_by_cond_pre[c] for c in conds_by_expected['unexpected'] if c in A_by_cond_pre])
    A_resps_all = pd.concat([A_resps_unexp, A_resps_exp])

    X_resps_exp = pd.concat([X_by_cond_pre[c] for c in conds_by_expected['expected'] if c in X_by_cond_pre])
    X_resps_unexp = pd.concat([X_by_cond_pre[c] for c in conds_by_expected['unexpected'] if c in X_by_cond_pre])
    X_resps_all = pd.concat([X_resps_unexp, X_resps_exp])

    # Extract baseline pre-windows (Pre pattern and Pre X)
    preA_data = A_resps_all.loc[:, -0.5:0].mean(axis=1)
    preX_data = X_resps_all.loc[:, -0.5:-0.01].mean(axis=1)

    # Panel L: Pre pattern boxplot
    fig_1m_plot_data = {}
    fig_l, ax_l = plt.subplots(figsize=(1.5, 1.8))
    for expected_val, pos in zip([1, 0], [0, 1]):
        subset = preA_data.xs(level='expected', key=expected_val) if 'expected' in preA_data.index.names else preA_data[A_resps_all['expected'] == expected_val]
        ax_l.boxplot(subset, positions=[pos],  **boxplot_kwargs)
        fig_1m_plot_data[f'A_{expected_val}_mean'] = subset.mean()
        fig_1m_plot_data[f'A_{expected_val}_sem'] = subset.sem()

    ax_l.set_xticks([0, 1])
    ax_l.set_xticklabels(['Exp', 'Unexp'])
    ax_l.set_ylabel('Pupil size (a.u)')
    ax_l.set_title('Pre pattern')
    format_axis(ax_l)
    fig_l.set_layout_engine('tight')
    fig_l.show()

    # Panel M: Pre X boxplot
    fig_m, ax_m = plt.subplots(figsize=(1.5, 1.8))
    for expected_val, pos in zip([1, 0], [0, 1]):
        subset = preX_data.xs(level='expected', key=expected_val) if 'expected' in preX_data.index.names else preX_data[X_resps_all['expected'] == expected_val]
        ax_m.boxplot(subset, positions=[pos], **boxplot_kwargs)
        fig_1m_plot_data[f'X_{expected_val}_mean'] = subset.mean()
        fig_1m_plot_data[f'X_{expected_val}_sem'] = subset.sem

    ax_m.set_xticks([0, 1])
    ax_m.set_xticklabels(['Exp', 'Unexp'])
    ax_m.set_ylabel('Pupil size (a.u)')
    ax_m.set_title('Pre X')
    format_axis(ax_m)
    fig_m.set_layout_engine('tight')
    fig_m.show()

    # =========================================================================
    # Panels O & P: Human Analysis
    # =========================================================================

    hum_A_by_cond_path = r"fig1_human_A_by_cond.h5"
    hum_X_by_cond_path = r"fig1_human_X_by_cond.h5"

    hum_A_by_cond = load_response_dfs(hum_A_by_cond_path)
    hum_X_by_cond = load_response_dfs(hum_X_by_cond_path)


    # Plot times series for human A and X responses
    hum_rare_freq_condnames = ['frequent', 'rare']
    hum_rf_sessions = np.intersect1d([hum_A_by_cond[c].index.get_level_values('sess').unique() for c in hum_rare_freq_condnames])
    fig_1o_plot_data = {}
    for stim_name, stim_by_cond in zip(['A', 'X'], [hum_A_by_cond, hum_X_by_cond]):
        hum_rare_freq_analysis = PupilCondAnalysis(stim_by_cond, hum_rare_freq_condnames, hum_rf_sessions)
        hum_rare_freq_analysis.plot_ts_by_cond(cond_line_kwargs=rare_freq_line_kwargs)
        hum_rare_freq_analysis.plots['ts_by_cond'][1].set_xlim(xlim)
        hum_rare_freq_analysis.plots['ts_by_cond'][0].set_size_inches(2, 2)
        hum_rare_freq_analysis.plots['ts_by_cond'][0].show()

        for cond, cond_resps in hum_rare_freq_analysis.by_cond.items():
            fig_1o_plot_data[f'{stim_name}_{cond}_mean'] = cond_resps.groupby('sess').mean().mean()
            fig_1o_plot_data[f'{stim_name}_{cond}_sem'] = cond_resps.groupby('sess').mean().sem()

    pd.DataFrame(fig_1o_plot_data).T.to_csv('fig_1o_human_rare_freq_plot_data.csv', index=True)

    # =========================================================================
    human_norm_dev_condnames = ['normal', 'deviant_C']
    fig_1p_plot_data = {}
    human_nd_sessions = ['Human28_220518', 'Human29_220523', 'Human30_220524', 'Human31_220530','Human32_220627']
    for stim_name, stim_by_cond in zip(['A', 'X'], [hum_A_by_cond, hum_X_by_cond]):
        hum_norm_dev_analysis = PupilCondAnalysis(stim_by_cond, human_norm_dev_condnames, human_nd_sessions)
        hum_norm_dev_analysis.plot_ts_by_cond(cond_line_kwargs=norm_dev_line_kwargs)
        hum_norm_dev_analysis.plots['ts_by_cond'][1].set_xlim(xlim)
        hum_norm_dev_analysis.plots['ts_by_cond'][0].set_size_inches(2, 2)
        hum_norm_dev_analysis.plots['ts_by_cond'][0].show() 


        for cond, cond_resps in hum_norm_dev_analysis.by_cond.items():
            fig_1p_plot_data[f'{stim_name}_{cond}_mean'] = cond_resps.groupby('sess').mean().mean()
            fig_1p_plot_data[f'{stim_name}_{cond}_sem'] = cond_resps.groupby('sess').mean().sem()

    pd.DataFrame(fig_1p_plot_data).T.to_csv('fig_1p_human_norm_dev_plot_data.csv', index=True)