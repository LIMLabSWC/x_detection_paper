from itertools import combinations
from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import ttest_rel, ttest_ind, ttest_1samp
from xdetectioncore.io_utils import load_response_dfs
from xdetectioncore.plotting import format_axis, set_nested_xlabels
from xdetectioncore.pupil.pupil_analysis_funcs import add_name_to_response_dfs
from xdetectioncore.behaviour import get_all_cond_filts
from pupil_analysis_funcs import PupilCondAnalysis

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


if __name__ == '__main__':

    plt.style.use('figure_stylesheet.mplstyle')

    ###################################################################################################################
    ##### MUSCIMOL ################################################################################################
    ###################################################################################################################
    # Data paths for Muscimol dataset
    A_by_cond_path = (
        DATA_DIR / "pupil_responses" / "fig02_muscimol_A_pupil_by_condition.h5"
    )
    X_by_cond_path = (
        DATA_DIR / "pupil_responses" / "fig02_muscimol_X_pupil_by_condition.h5"
    )
    musc_td_df_path = DATA_DIR / "trial_tables" / "fig02_muscimol_trials.h5"
    musc_sess_dict_path = (
        DATA_DIR / "session_metadata" / "fig02_muscimol_sessions_by_state.json"
    )

    # Load condition pupil response DataFrames with unique panel-specific variable names
    A_musc_by_cond = load_response_dfs(A_by_cond_path)
    X_musc_by_cond = load_response_dfs(X_by_cond_path)
    for _by_cond in [A_musc_by_cond, X_musc_by_cond]:
        add_name_to_response_dfs(_by_cond)

    # Load trial data
    musc_td_df = pd.read_hdf(musc_td_df_path)
    assert isinstance(musc_td_df, pd.DataFrame)

    # Load drug session dictionary
    with open(musc_sess_dict_path, 'r') as f:
        drug_sess_dict = json.load(f)

    cond_filters = get_all_cond_filts()
    good_trial_filt = 'drug_key <=5 & Trial_Outcome == 1'

    rare_freq_line_kwargs = {
        'rare': {'c': '#00cbfcff', 'ls': '-'},
        'frequent': {'c': '#1530e9ff', 'ls': '-'}
    }

    xlim = [-0.25, 2.25]

    # Filter out excluded dates and DO76 from drug sessions
    exclude_dates = ['240710', '240712', '240716', '240717', '240726']

    none_sess = [
        sess for sess in drug_sess_dict.get('none', [])
        if sess.split('_')[1] not in exclude_dates
           # and 'DO76' not in sess
           # and 'DO71' not in sess
    ]
    musc_sess = [
        sess for sess in drug_sess_dict.get('muscimol', [])
        if sess.split('_')[1] not in exclude_dates
           # and 'DO76' not in sess
           # and 'DO71' not in sess
    ]
    sess_2_use_by_drug ={'none':none_sess, 'muscimol':musc_sess}

    drugs_to_plot = [('none', 'Control'), ('muscimol', 'Muscimol')]

    boxplot_kwargs = dict(
        widths=0.4,
        patch_artist=False,
        showmeans=False,
        showfliers=False,
        medianprops=dict(mfc='k', color='k'),
        boxprops=dict(lw=0.5),
        whiskerprops=dict(lw=0.5),
        capprops=dict(lw=0.5),
    )

    # =========================================================================
    # 1. Peak PDR Analysis (Rare vs Frequent across Drug States)
    # =========================================================================
    rare_freq_condnames = ['frequent', 'rare']
    rare_freq_by_drug_by_cond = {}

    for drug_key, drug_label in drugs_to_plot:
        drug_sess_list = sess_2_use_by_drug[drug_key]

        # Extract peak PDR in window [0.5, 2.5]s after smoothing
        resps_by_cond = {
            cond: A_musc_by_cond[cond].T.rolling(window=25).mean().T
                  .loc[A_musc_by_cond[cond].index.get_level_values('sess').isin(drug_sess_list), 0.5:2.5]
            .max(axis=1).groupby(['name']).mean()
            for cond in rare_freq_condnames
        }

        cond_df = pd.DataFrame.from_dict(resps_by_cond).dropna()
        rare_freq_by_drug_by_cond[drug_key] = cond_df

    # --- Figure: Combined Peak PDR Comparison Matching Reference Layout ---
    fig, ax = plt.subplots(figsize=(2.2, 2.5))

    # Positions for: [Frequent Control, Frequent Muscimol, Rare Control, Rare Muscimol]
    x_positions = [1.0, 1.8, 3.2, 4.0]

    plot_data = [
        rare_freq_by_drug_by_cond['none']['frequent'].values,
        rare_freq_by_drug_by_cond['muscimol']['frequent'].values,
        rare_freq_by_drug_by_cond['none']['rare'].values,
        rare_freq_by_drug_by_cond['muscimol']['rare'].values,
    ]

    # Style boxes with white fill and dark red median lines matching reference style
    custom_boxplot_kwargs = dict(
        widths=0.6,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        boxprops=dict(facecolor='w', edgecolor='k', lw=0.5),
        medianprops=dict(color='darkred', lw=1.5),
        whiskerprops=dict(lw=0.5),
        capprops=dict(lw=0.5),
    )

    box_plot = ax.boxplot(plot_data, positions=x_positions, **custom_boxplot_kwargs)

    format_axis(ax)
    ax.set_xticks([1.4, 3.6])
    ax.set_xticklabels(['Frequent', 'Rare'])
    ax.set_ylabel('Max pupil size (a.u)')
    ax.set_title('Peak PDR by condition', fontsize=9)

    # Add secondary sub-labels underneath for Control vs Muscimol per condition group
    ax.text(1.0, -0.15, 'Control', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=7)
    ax.text(1.8, -0.15, 'Muscimol', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=7)
    ax.text(3.2, -0.15, 'Control', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=7)
    ax.text(4.0, -0.15, 'Muscimol', ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=7)

    fig.set_layout_engine('constrained')
    fig.show()

    pd.DataFrame(plot_data, index=['Frequent Control', 'Frequent Muscimol', 'Rare Control', 'Rare Muscimol']).T.to_csv('fig2_peak_pdr_boxplot_data.csv')

    # Peak PDR Statistical Comparisons
    ttest_ctrl = ttest_rel(rare_freq_by_drug_by_cond['none']['rare'], rare_freq_by_drug_by_cond['none']['frequent'])
    ttest_musc = ttest_rel(rare_freq_by_drug_by_cond['muscimol']['rare'],
                           rare_freq_by_drug_by_cond['muscimol']['frequent'])

    ctrl_diff = rare_freq_by_drug_by_cond['none']['rare'] - rare_freq_by_drug_by_cond['none']['frequent']
    musc_diff = rare_freq_by_drug_by_cond['muscimol']['rare'] - rare_freq_by_drug_by_cond['muscimol']['frequent']
    ttest_diff = ttest_ind(ctrl_diff.dropna(), musc_diff.dropna(), alternative='greater')

    print(f"Control Peak PDR (Rare vs Freq) paired t-test: {ttest_ctrl}")
    print(f"Muscimol Peak PDR (Rare vs Freq) paired t-test: {ttest_musc}")
    print(f"Control vs Muscimol Delta Peak PDR independent t-test: {ttest_diff}")

    # --- Figure: Delta Peak PDR (Rare - Frequent) Boxplot ---
    fig, ax = plt.subplots(figsize=(1.2, 2))

    diff_data = [ctrl_diff.dropna().values, musc_diff.dropna().values]
    ax.boxplot(diff_data, labels=['Control', 'Muscimol'], **boxplot_kwargs)

    for i, d in enumerate(diff_data):
        ax.scatter([i + 1] * len(d), d, fc='k', alpha=0.5, s=15, ec='none')

    ax.axhline(0, color='k', ls='--', lw=0.5)
    format_axis(ax)
    ax.set_ylabel('Δ Peak PDR (Rare - Freq)')
    fig.set_layout_engine('constrained')
    fig.show()

    pd.DataFrame({'Control': ctrl_diff.dropna().values, 'Muscimol': musc_diff.dropna().values}).to_csv('fig2_delta_peak_pdr_boxplot_data.csv')

    # =========================================================================
    # 2. X Response Plot (hit_all vs miss_all)
    # =========================================================================
    x_sess_list = X_musc_by_cond['miss_all'].index.get_level_values('sess').unique()
    X_resp = PupilCondAnalysis(
        by_cond_dict=X_musc_by_cond,
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
    # X_resp.plots['ts_by_cond'][0].show()

    # =========================================================================
    # 3. Rare vs Frequent Time-Series Plot (Control vs Muscimol)
    # =========================================================================
    musc_resp_plot_data_by_stim = {}
    musc_by_drug_sess = {}
    mutual_sessions = [s for s in A_musc_by_cond['rare'].index.get_level_values('sess')
                       if s in A_musc_by_cond['frequent'].index.get_level_values('sess')]
    for stim_name, stim_by_cond in zip(['pattern', 'X'], [A_musc_by_cond, X_musc_by_cond]):
        musc_resp_plot_data = []
        for drug_key, drug_label in drugs_to_plot:
            target_sess = [s for s in sess_2_use_by_drug[drug_key] if s in mutual_sessions]
            if not target_sess:
                continue
            rf_analysis = PupilCondAnalysis(stim_by_cond, rare_freq_condnames, target_sess,
                                            event_name=stim_name)
            rf_analysis.plot_ts_by_cond(cond_line_kwargs=rare_freq_line_kwargs)
            musc_by_drug_sess[f'{drug_label}'] = rf_analysis.by_cond['rare'].index.get_level_values('sess').unique()

            ax = rf_analysis.plots['ts_by_cond'][1]
            ax.set_xlim(xlim)
            if stim_name == 'pattern':
                ax.set_ylim(-0.05, 0.32)
            else:
                ax.set_ylim(-0.05, 0.7)
            ax.set_title(f'{drug_label}\nPDR to {stim_name}')

            fig = rf_analysis.plots['ts_by_cond'][0]
            fig.set_size_inches(1.5, 2)
            fig.show()
            musc_resp_plot_data.append(rf_analysis.by_cond)
        for di, d_resps in enumerate(musc_resp_plot_data):
            for cond, resps in d_resps.items():
                musc_resp_plot_data_by_stim[f'PDR_mean_{stim_name}_{cond}_{drugs_to_plot[di][0]}'] = resps.groupby('sess').mean().mean(axis=0)
                musc_resp_plot_data_by_stim[f'PDR_sem_{stim_name}_{cond}_{drugs_to_plot[di][0]}'] = resps.groupby('sess').mean().sem(axis=0)
                print(f'{stim_name} {cond}: {resps.groupby("sess").mean().shape}')
    musc_plot_data_df = pd.DataFrame(musc_resp_plot_data_by_stim)
    musc_plot_data_df.T.to_csv('fig2b_musc_rare_freq_ts_plot_data.csv')

    # =========================================================================
    # 4. Behavioral Performance Summary Across Tone Positions
    # =========================================================================
    patt_nonpatt_cols = ['dimgray', 'indigo']

    # colate performance data for each drug state and plot
    for drug_key, drug_label in drugs_to_plot:
        drug_sess_list = [sess for sess in drug_sess_dict.get(drug_key, [])
                          if 'DO76' not in sess
                          and 'DO71' not in sess]
        rare_freq_sess_df = musc_td_df.loc[musc_td_df.index.get_level_values('sess').isin(drug_sess_list)]

        perf_data = (
            rare_freq_sess_df.query('Stage>=3 & 20<trial_num<=250')
            .groupby(['name', 'Tone_Position'])['Trial_Outcome']
            .mean()
            .unstack()
            .dropna(axis=0)[[1, 0]]
            .values
        )

        fig, ax = plt.subplots(figsize=(1.25, 1.5))
        ax.bar(['Non pattern', 'Pattern'], perf_data.mean(axis=0), ec='black', fc='w', lw=0.5, width=0.4)
        for cond_idx, (cond_label, color) in enumerate(zip(['Non pattern', 'Pattern'], patt_nonpatt_cols)):
            ax.scatter([cond_label] * len(perf_data[:, cond_idx]), perf_data[:, cond_idx], facecolor=color,
                       alpha=0.5, lw=0.01)

        ax.set_ylim(0.4, 1.01)
        format_axis(ax)
        ax.set_ylabel('Hit rate')
        ax.set_title(f'{drug_label}')
        fig.set_layout_engine('constrained')
        # fig.show()

    # =========================================================================
    # 5. Overall Performance & Reaction Time Summary Across Tone Positions
    # =========================================================================
    patt_nonpatt_cols = ['dimgray', 'indigo']

    for drug_key, drug_label in drugs_to_plot:
        drug_sess_list = [sess for sess in drug_sess_dict.get(drug_key, []) if 'DO76' not in sess]
        rare_freq_sess_df = musc_td_df.loc[musc_td_df.index.get_level_values('sess').isin(drug_sess_list)]

        perf_data = (
            rare_freq_sess_df.query('Stage>=3 & 20<trial_num<=250')
            .groupby(['name', 'Tone_Position'])['Trial_Outcome']
            .mean()
            .unstack()
            .dropna(axis=0)[[1, 0]]
            .values
        )

        fig, ax = plt.subplots()
        ax.bar(['Non pattern', 'Pattern'], perf_data.mean(axis=0), ec='black', fc='w', lw=0.5, width=0.4)
        for cond_idx, (cond_label, color) in enumerate(zip(['Non pattern', 'Pattern'], patt_nonpatt_cols)):
            ax.scatter([cond_label] * len(perf_data[:, cond_idx]), perf_data[:, cond_idx], facecolor=color,
                       alpha=0.5, lw=0.01)
        for a_data in perf_data:
            ax.plot(['Non pattern', 'Pattern'], a_data, lw=0.25, color='gray')

        ax.set_ylim(0.4, 1.01)
        format_axis(ax)
        ax.set_ylabel('Hit rate')
        ax.set_title(f'{drug_label}')
        fig.set_size_inches(1.25, 1.5)
        fig.set_layout_engine('constrained')

    # =========================================================================
    # Muscimol Behavioral Performance (Hit Rate) Bar Plot
    # =========================================================================

    musc_states_to_plot = [('none', 'Control'), ('muscimol', 'Muscimol')]
    musc_cond_names = ['frequent', 'rare']

    fig_musc_perf, ax_musc_perf = plt.subplots(figsize=(2.2, 2.5))

    x_positions_musc = [0.8, 1.6, 3.0, 3.8]
    musc_bar_means = []
    musc_bar_indiv = []
    musc_animals = ['DO75', 'DO72', 'DO83', 'DO85', 'DO84', 'DO86']

    for drug_key, _ in musc_states_to_plot:
        drug_sess_list = [sess for sess in drug_sess_dict.get(drug_key, [])
                          if any([n in sess for n in musc_animals])]
        state_td_df = musc_td_df.loc[musc_td_df.index.get_level_values('sess').isin(drug_sess_list)]

        for cond in musc_cond_names:
            print(f'{drug_key}, {cond}')
            perf_by_name = (
                state_td_df.query('Stage==3 & 30 <trial_num<=250')
                # state_td_df.query('Stage==3 & n_since_last_Trial_Outcome <=5')
                .query(cond_filters[cond])['Trial_Outcome']
                .groupby('name')
                .mean()
            )
            musc_bar_means.append(perf_by_name.mean())
            musc_bar_indiv.append(perf_by_name.values)

    ax_musc_perf.bar(
        x_positions_musc,
        musc_bar_means,
        width=0.6,
        facecolor='w',
        edgecolor='k',
        lw=0.5
    )
    pd.DataFrame(musc_bar_indiv, index=['Control Frequent', 'Control Rare', 'Muscimol Frequent', 'Muscimol Rare']).T.to_csv('fig2_musc_behavioral_performance_bar_plot_data.csv')

    scat_colors_musc = ['#1530e9ff', '#00cbfcff', '#1530e9ff', '#00cbfcff']
    for idx, vals in enumerate(musc_bar_indiv):
        ax_musc_perf.scatter(
            [x_positions_musc[idx]] * len(vals),
            vals,
            color=scat_colors_musc[idx],
            alpha=0.6,
            s=12,
            edgecolor='none'
        )

    ax_musc_perf.set_ylim(0.4, 1.15)
    ax_musc_perf.set_ylabel('Hit rate')
    format_axis(ax_musc_perf)

    ax_musc_perf.set_xticks(x_positions_musc)
    ax_musc_perf.set_xticklabels(['Frequent', 'Rare', 'Frequent', 'Rare'], fontsize=7)

    ax_musc_perf.text(1.2, 0.28, 'Control', ha='center', va='top', transform=ax_musc_perf.get_xaxis_transform(),
                      fontsize=8)
    ax_musc_perf.text(3.4, 0.28, 'Muscimol', ha='center', va='top', transform=ax_musc_perf.get_xaxis_transform(),
                      fontsize=8)

    all_combs = [f'{d}, {c}' for d in ['control', 'muscimol'] for c in musc_cond_names]
    ttest_by_comb = []
    for comb in list(combinations(range(len(musc_bar_means)), 2)):
        print(f'{all_combs[comb[0]]} vs {all_combs[comb[1]]}')
        res = ttest_rel(musc_bar_indiv[comb[0]], musc_bar_indiv[comb[1]])
        print(res)
        ttest_by_comb.append(res.pvalue)

    def draw_bracket(ax, x1, x2, y, text):
        h = 0.02
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='k', lw=0.5)
        ax.text((x1 + x2) * 0.5, y + h, text, ha='center', va='bottom', fontsize=7)


    draw_bracket(ax_musc_perf, 0.8, 1.6, 1.02, 'ns' if ttest_by_comb[0] > 0.05 else '*')
    draw_bracket(ax_musc_perf, 3.0, 3.8, 1.02, 'ns' if ttest_by_comb[3] > 0.05 else '*')
    draw_bracket(ax_musc_perf, 1.2, 3.4, 1.10, 'ns' if ttest_by_comb[2] > 0.05 else '*')

    fig_musc_perf.set_layout_engine('constrained')
    fig_musc_perf.show()
    fig_musc_perf.savefig('musc_behavioral_performance_bar_plot.pdf', format='pdf')



    # --- B. Reaction Time (RT) Bar Plot ---
    fig_musc_rt, ax_musc_rt = plt.subplots(figsize=(2.0, 2.2))
    musc_rt_means, musc_rt_indiv = [], []
    musc_animals = ['DO75', 'DO72', 'DO83', 'DO85', 'DO84', 'DO86']


    for drug_key, _ in musc_states_to_plot:
        drug_sess_list = [sess for sess in drug_sess_dict.get(drug_key, [])
                          if any([n in sess for n in musc_animals])]
        state_td_df = musc_td_df.loc[musc_td_df.index.get_level_values('sess').isin(drug_sess_list)]
        # correct trials only
        state_td_df = state_td_df.loc[state_td_df['Trial_Outcome']==1]

        for cond in musc_cond_names:
            cond_td_df = state_td_df.query('Stage==3 & 30 <trial_num<=250 & Trial_Outcome==1').query(cond_filters[cond])
            rt = (cond_td_df['Trial_End_dt'] - cond_td_df['Gap_Time_dt']).dt.total_seconds()
            rt_by_name = rt.groupby('name').mean()
            musc_rt_means.append(rt_by_name.mean())
            musc_rt_indiv.append(rt_by_name.values)

    ax_musc_rt.bar(x_positions_musc, musc_rt_means, width=0.35, facecolor='w', edgecolor='k', lw=0.5)
    for idx, vals in enumerate(musc_rt_indiv):
        ax_musc_rt.scatter([x_positions_musc[idx]] * len(vals), vals, color=scat_colors_musc[idx], alpha=0.6, s=12,
                           edgecolor='none')

    ax_musc_rt.set_ylabel('Reaction time (s)')
    format_axis(ax_musc_rt)
    ax_musc_rt.set_xticks(x_positions_musc)
    ax_musc_rt.set_xticklabels(['Frequent', 'Rare', 'Frequent', 'Rare'], fontsize=7)
    ax_musc_rt.text(0.5, -0.22, 'Control', ha='center', va='top', transform=ax_musc_rt.get_xaxis_transform(),
                    fontsize=8)
    ax_musc_rt.text(3.0, -0.22, 'Muscimol', ha='center', va='top', transform=ax_musc_rt.get_xaxis_transform(),
                    fontsize=8)
    fig_musc_rt.set_layout_engine('constrained')
    fig_musc_rt.show()
    # fig_musc_rt.savefig('musc_reaction_time_bar_plot.pdf', format='pdf')

    # =========================================================================
    # Opto (Normal vs Deviant ABAD)
    # =========================================================================

    A_by_cond_path = (
        DATA_DIR / "pupil_responses" / "fig02_opto_A_pupil_by_condition.h5"
    )
    X_by_cond_path = (
        DATA_DIR / "pupil_responses" / "fig02_opto_X_pupil_by_condition.h5"
    )
    opto_td_df_path = DATA_DIR / "trial_tables" / "fig02_opto_normdev_trials.h5"
    opto_sess_dict_path = (
        DATA_DIR / "session_metadata" / "fig02_fig04_opto_sessions_by_state.json"
    )

    # Load condition pupil response DataFrames with unique opto variable name
    A_opto_by_cond = load_response_dfs(A_by_cond_path)
    X_opto_by_cond = load_response_dfs(X_by_cond_path)

    # Load trial data
    # all_batch = [f'RS0{i}' for i in '2378']
    all_batch = [f'RS0{i}' for i in '123678']

    opto_td_df = pd.read_hdf(opto_td_df_path)
    assert isinstance(opto_td_df, pd.DataFrame)
    opto_td_df = opto_td_df.loc[opto_td_df.index.get_level_values('name').isin(all_batch)]

    # Load opto state session dictionary
    with open(opto_sess_dict_path, 'r') as f:
        opto_sess_dict = json.load(f)

    cond_filters = get_all_cond_filts()
    good_trial_filt = 'n_since_last_Trial_Outcome <=5 '

    xlim = [-0.25, 2.25]

    # =========================================================================
    # 3. Normal vs Deviant Analysis Across Opto States
    # =========================================================================
    norm_dev_line_kwargs = {
        'normal': {'c': 'black', 'ls': '-'},
        'deviant_C': {'c': '#cd2727ff', 'ls': '-'}
    }
    norm_dev_cond_names = ['normal', 'deviant_C']

    norm_dev_sess = [
        sess for sess in A_opto_by_cond['normal'].index.get_level_values('sess').unique()
        if 'deviant_C' in A_opto_by_cond and sess in A_opto_by_cond['deviant_C'].index.get_level_values('sess')
    ]

    # Time-series analysis across conditions & stimuli
    diff_ts_plot = plt.subplots()
    for stim_name, stim_by_cond in zip(['A', 'X'], [A_opto_by_cond, X_opto_by_cond]):

        for opto_state in ['light_off', 'learning_block_light_on']:
            target_sess = [
                sess for sess in norm_dev_sess
                if sess in opto_sess_dict.get(opto_state, [])
                   and sess.split('_')[0] in all_batch
            ]
            n_sess = 4
            state_sess_list = [sorted([sess for sess in target_sess if name in sess])[:n_sess]
                               for name in all_batch]
            state_sess_list = sum(state_sess_list, [])

            print(f'{opto_state}: {len(state_sess_list)}')
            nd_analysis = PupilCondAnalysis(stim_by_cond, norm_dev_cond_names, state_sess_list)
            nd_analysis.plot_ts_by_cond(cond_line_kwargs=norm_dev_line_kwargs)
            nd_analysis.plots['ts_by_cond'][1].set_xlim(xlim)
            nd_analysis.plots['ts_by_cond'][1].set_ylim((-0.15, 0.4) if stim_name == 'A' else (-.05,1))

            nd_analysis.plots['ts_by_cond'][1].set_title(f'{stim_name} Norm/Dev ({opto_state})')
            nd_analysis.plots['ts_by_cond'][0].set_size_inches(2, 2)
            nd_analysis.plots['ts_by_cond'][0].show()
            if stim_name == 'A':
                nd_analysis.plot_diff_ts_by_cond(group_level='name',plot=diff_ts_plot)

    diff_ts_plot[1].set_xlim(0,2)
    diff_ts_plot[0].show()

    # Behavioral performance: Normal vs Deviant
    cond_td_df = opto_td_df.loc[opto_td_df.index.get_level_values('sess').isin(norm_dev_sess)]
    norm_dev_perf = [
        cond_td_df.query(good_trial_filt).query(cond_filters[cond])['Trial_Outcome'].groupby('name').mean()
        for cond in norm_dev_cond_names
        if cond in cond_filters
    ]

    fig, ax = plt.subplots()
    ax.bar(norm_dev_cond_names, [e.mean() for e in norm_dev_perf], ec='black', fc='w', lw=0.5, width=0.4)
    for cond, data in zip(norm_dev_cond_names, norm_dev_perf):
        ax.scatter([cond] * len(data), data, edgecolor='none', fc=norm_dev_line_kwargs[cond]['c'], alpha=0.5)

    format_axis(ax)
    fig.set_size_inches(1.5, 2)
    fig.set_layout_engine('constrained')
    fig.show()

    # =========================================================================
    # Opto Behavioral Performance Bar Plot (Normal vs Deviant)
    # =========================================================================
    all_batch = [f'RS0{i}' for i in '123678']

    # opto_sess_dict['light_on'] = opto_sess_dict['light_on']+opto_sess_dict['learning_block_light_on'] + opto_sess_dict['testing_block_light_on']

    opto_states_to_plot = [
        ('light_off', 'Control'),
        ('light_on', 'Inhibition')
    ]
    opto_cond_names = ['normal', 'deviant_C']

    fig_opto_perf, ax_opto_perf = plt.subplots(figsize=(2.0, 2.2))

    bar_width = 0.35
    x_positions_opto = [0, 1, 2.5, 3.5]
    opto_bar_means = []
    opto_bar_indiv = []

    for opto_state, _ in opto_states_to_plot:
        state_sess_list = opto_sess_dict.get(opto_state, [])
        state_sess_list = [s for s in state_sess_list]


        state_td_df = opto_td_df.loc[opto_td_df.index.get_level_values('sess').isin(state_sess_list)]

        for cond in opto_cond_names:
            perf_by_name = (
                state_td_df.query(cond_filters[cond])['Trial_Outcome']
                # state_td_df.query('Tone_Position==0 & Stage==4 & Pattern_Type==1 & N_TonesPlayed==4')['Trial_Outcome']
                .groupby('name')
                .mean()
            )
            opto_bar_means.append(perf_by_name.values[-5:].mean())
            opto_bar_indiv.append(perf_by_name.values[-5:])

    ax_opto_perf.bar(
        x_positions_opto,
        opto_bar_means,
        width=bar_width,
        facecolor='w',
        edgecolor='k',
        lw=0.5
    )

    scat_colors_opto = ['k', '#cd2727ff'] * 2
    for idx, vals in enumerate(opto_bar_indiv):
        ax_opto_perf.scatter(
            [x_positions_opto[idx]] * len(vals),
            vals,
            color=scat_colors_opto[idx],
            alpha=0.6,
            s=12,
            edgecolor='none'
        )

    ax_opto_perf.set_ylim(0.0, 1.05)
    ax_opto_perf.set_ylabel('Hit rate')
    format_axis(ax_opto_perf)

    ax_opto_perf.set_xticks(x_positions_opto)
    ax_opto_perf.set_xticklabels(['Norm', 'Dev', 'Norm', 'Dev'], fontsize=8)

    ax_opto_perf.text(0.5, -0.22, 'Control', ha='center', va='top', transform=ax_opto_perf.get_xaxis_transform(),
                      fontsize=8)
    ax_opto_perf.text(3.0, -0.22, 'Inhibition', ha='center', va='top', transform=ax_opto_perf.get_xaxis_transform(),
                      fontsize=8)

    fig_opto_perf.set_layout_engine('constrained')
    fig_opto_perf.show()

    pd.DataFrame(opto_bar_indiv, index=['Control Normal', 'Control Deviant', 'Inhibition Normal', 'Inhibition Deviant']).T.to_csv('fig2_opto_behavioral_performance_bar_plot_data.csv')

    # Calculate paired t-tests
    p_ctrl = ttest_rel(opto_bar_indiv[0], opto_bar_indiv[1]).pvalue
    p_inh = ttest_rel(opto_bar_indiv[2], opto_bar_indiv[3]).pvalue
    p_dev = ttest_rel(opto_bar_indiv[1], opto_bar_indiv[3]).pvalue

    draw_bracket(ax_opto_perf, 0.8, 1.6, 1.02, 'ns' if p_ctrl > 0.05 else '*')
    draw_bracket(ax_opto_perf, 3.0, 3.8, 1.02, 'ns' if p_inh > 0.05 else '*')
    draw_bracket(ax_opto_perf, 1.2, 3.4, 1.10, 'ns' if p_dev > 0.05 else '*')


    # =========================================================================
    # Opto Normal vs rule violating/conforming deviants
    # =========================================================================

    # Path to pre-saved HDF5 file containing all subset responses
    subset_resps_path = (
        DATA_DIR / "pupil_responses" / "fig02_opto_subset_A_pupil_responses.h5"
    )

    all_subset_resps = load_response_dfs(subset_resps_path)

    df_to_save = []
    for subset_name, subset_df in all_subset_resps.items():
        if not isinstance(subset_df, pd.DataFrame):
            continue

        resps_by_cond = subset_df.groupby(['sess', 'condition']).mean().groupby('condition').mean()
        resps_by_cond.index = resps_by_cond.index = resps_by_cond.index.astype(str) + f'_{subset_name}_mean'

        resps_sems_by_cond = subset_df.groupby(['sess', 'condition']).mean().groupby('condition').sem()
        resps_sems_by_cond.index = resps_sems_by_cond.index = resps_sems_by_cond.index.astype(str) + f'_{subset_name}_sem'
        df_to_save.append(pd.concat([resps_by_cond, resps_sems_by_cond], axis=0))

    df_to_save = pd.DataFrame(pd.concat(df_to_save,axis=0))
    df_to_save.to_csv('fig2j_opto_rule_violation_ts_plot_data.csv')

    dev_cond_names = ['normal', 'dev_ABCD1', 'dev_ABBA1']
    dev_line_kwargs = {
        'normal': {'c': 'k', 'ls': '-'},
        'dev_ABCD1': {'c': '#e8739bff', 'ls': '-'},
        'dev_ABBA1': {'c': '#73e8c1ff', 'ls': '-'}
    }
    plt.style.use('figure_stylesheet.mplstyle')

    dev_diff_by_subset = {}
    mean_pupil_by_subset = {}


    # --- Plot mean time series responses ---
    for subset_name, subset_df in all_subset_resps.items():
        assert isinstance(subset_df, pd.DataFrame)

        by_cond = {
            cond: subset_df.xs(cond, level='condition')
            for cond in dev_cond_names
            if cond in subset_df.index.get_level_values('condition')
        }

        analysis_obj = PupilCondAnalysis(by_cond_dict=by_cond, conditions=dev_cond_names)
        analysis_obj.plot_ts_by_cond(cond_line_kwargs=dev_line_kwargs)
        analysis_obj.plots['ts_by_cond'][1].set_title(subset_name)
        analysis_obj.plots['ts_by_cond'][0].show()

    # --- Difference Across Windows Analysis ---


    for subset_name, subset_df in all_subset_resps.items():
        assert isinstance(subset_df, pd.DataFrame)
        by_cond = {
            cond: subset_df.xs(cond, level='condition')
            for cond in dev_cond_names
            if cond in subset_df.index.get_level_values('condition')
        }

        pupil_by_cond = [
            by_cond[cond].subtract(by_cond[cond].loc[:, -1:0].mean(axis=1), axis=0)
            for cond in dev_cond_names
        ]
        pupil_by_cond = [
            cond_resps.loc[:, 0.7:1.25].mean(axis=1).groupby(['sess', 'name']).mean()
            for cond_resps in pupil_by_cond
        ]

        mean_pupil_by_subset[subset_name] = {k: v for k, v in zip(dev_cond_names, pupil_by_cond)}

        diff_across_windows_by_state = {
            'rule': [
                (pupil_by_cond[i] - pupil_by_cond[0]).groupby('name').mean()
                for i, _ in enumerate(pupil_by_cond)
            ]
        }
        dev_diff_across_windows_by_state = (pupil_by_cond[2] - pupil_by_cond[1]).groupby('name').mean()
        dev_diff_by_subset[subset_name] = dev_diff_across_windows_by_state

        plt_cols = ['k', '#0066ffe6']
        fig, ax = plt.subplots()
        diffs_as_dfs = [pd.concat(e, axis=1).loc[:, 1:] for e in diff_across_windows_by_state.values()]

        for ii, _df in enumerate(diffs_as_dfs):
            ax.errorbar(
                _df.columns + ii * 0.05,
                _df.mean(axis=0),
                yerr=_df.sem(axis=0),
                fmt='.',
                lw=2,
                alpha=0.9,
                markersize=15,
                c=plt_cols[ii]
            )

        ax.axhline(0, color='k', ls='--')
        ax.set_ylim(0.0, 0.68)
        ax.locator_params(axis='y', nbins=4)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Conforming', 'Violating'])

        fig.set_size_inches(1.5, 2)
        fig.show()
        fig.savefig(f'rule_diff_{subset_name}.pdf')

        print(subset_name, ttest_rel(diffs_as_dfs[0][1], diffs_as_dfs[0][2], alternative='less'))

    # --- Summary Comparison Plot Across Subsets ---
    fig, ax = plt.subplots()

    # --- Summary Comparison Plot Across Subsets (Control vs Inhibition) ---
    rule_resps_by_subset_df = {
        k: pd.DataFrame.from_dict(v, orient='columns')
        for k, v in mean_pupil_by_subset.items()
    }

    fig_l_sum, ax_l_sum = plt.subplots(figsize=(1.8, 2.5))

    summary_subsets = [('control', 'Control'), ('opto', 'Inhibition')]
    rule_cols = [
        ('dev_ABCD1', 'rule conforming', '#e8739bff'),
        ('dev_ABBA1', 'rule violating', '#73e8c1ff'),
    ]

    x_centers = [0, 1]
    offset = 0.14

    for s_idx, (subset_key, _) in enumerate(summary_subsets):
        sub_df = (rule_resps_by_subset_df[subset_key] - rule_resps_by_subset_df[subset_key][['normal']].values)
        sub_by_name = sub_df.groupby('name').mean()

        for c_idx, (cond_key, label_txt, col_hex) in enumerate(rule_cols):
            x_pos = x_centers[s_idx] + (-offset if c_idx == 0 else offset)
            vals = sub_by_name[cond_key].dropna()
            ax_l_sum.errorbar(
                x_pos,
                vals.mean(),
                yerr=vals.sem(),
                fmt='o',
                lw=2,
                markersize=7,
                c=col_hex,
                ecolor=col_hex,
                label=label_txt if s_idx == 0 else None,
            )

    ax_l_sum.set_xticks(x_centers)
    ax_l_sum.set_xticklabels([label for _, label in summary_subsets])
    ax_l_sum.set_xlim(-0.45, 1.45)
    ax_l_sum.set_yticks([0.2, 0.4, 0.6])
    ax_l_sum.set_ylabel('Δ Pupil size (a.u.)')
    format_axis(ax_l_sum)
    ax_l_sum.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=1)

    fig_l_sum.set_layout_engine('constrained')
    fig_l_sum.show()

    if 'opto' in rule_resps_by_subset_df and 'post_opto' in rule_resps_by_subset_df:
        for cond in dev_cond_names[1:]:
            opto_vals = rule_resps_by_subset_df['opto'][cond].groupby('name').mean()
            post_vals = rule_resps_by_subset_df['post_opto'][cond].groupby('name').mean()
            print(f"{cond}: {ttest_rel(opto_vals, post_vals)}")
    

    # =========================================================================
    # Paired / One-Sample t-test on Rule Violation Difference Scores
    # d_i = (violating - conforming)_control - (violating - conforming)_inhibition
    # =========================================================================

    assert 'control' in rule_resps_by_subset_df and 'opto' in rule_resps_by_subset_df

    diff_control = (
            rule_resps_by_subset_df['control']['dev_ABBA1']
            - rule_resps_by_subset_df['control']['dev_ABCD1']
    ).groupby('name').mean()
    
    diff_inhibition = (
            rule_resps_by_subset_df['opto']['dev_ABBA1']
            - rule_resps_by_subset_df['opto']['dev_ABCD1']
    ).groupby('name').mean()

    common_mice = diff_control.index.intersection(diff_inhibition.index)
    diff_control = diff_control.loc[common_mice]
    diff_inhibition = diff_inhibition.loc[common_mice]

    d_i = diff_control - diff_inhibition

    t_stat, p_val = ttest_1samp(d_i, popmean=0)

    print("\n--- One-Sample t-test on Rule Violation Difference Scores (d_i) ---")
    print(f"Mice included (N={len(d_i)}): {list(d_i.index)}")
    print(f"d_i per mouse:\n{d_i}")
    print(f"Mean d_i: {d_i.mean():.4f} +/- {d_i.sem():.4f} (SEM)")
    print(f"t({len(d_i) - 1}) = {t_stat:.4f}, p = {p_val:.4f}")

    rule_diff_plot_data = pd.DataFrame({
        'diff_control': diff_control,
        'diff_inhibition': diff_inhibition,

    })
    rule_diff_plot_data.to_csv('fig2l_rule_violation_diff_plot_data.csv')

    # =========================================================================
    # Panel C Extensions: Baseline Pupil (Pre pattern, Pre X) & Behavior by Drug State
    # =========================================================================

    # 1. Load pre-computed condition response dictionaries for Panel C (using direct path without paper_data_dir)
    A_panel_c_pre_by_cond = load_response_dfs(
        DATA_DIR / "pupil_responses" / "fig02_muscimol_A_prestim_by_condition.h5"
    )
    X_panel_c_pre_by_cond = load_response_dfs(
        DATA_DIR / "pupil_responses" / "fig02_muscimol_X_prestim_by_condition.h5"
    )

    conds_by_expected_musc = {
        'expected': ['frequent'],
        'unexpected': ['rare'],
    }

    A_pre_musc_dict = {}
    X_pre_musc_dict = {}

    for cond_name, cond_list in conds_by_expected_musc.items():
        exp_val = 1 if cond_name == 'expected' else 0
        for c in cond_list:
            if c in A_panel_c_pre_by_cond:
                df_a = A_panel_c_pre_by_cond[c].copy()
                df_a['expected'] = exp_val
                df_a.set_index('expected', append=True, inplace=True)
                A_pre_musc_dict[c] = df_a
            if c in X_panel_c_pre_by_cond:
                df_x = X_panel_c_pre_by_cond[c].copy()
                df_x['expected'] = exp_val
                df_x.set_index('expected', append=True, inplace=True)
                X_pre_musc_dict[c] = df_x

    A_pre_resps_musc = pd.concat(list(A_pre_musc_dict.values()))
    X_pre_resps_musc = pd.concat(list(X_pre_musc_dict.values()))

    preA_data = A_pre_resps_musc.loc[:, -0.5:-0.01].mean(axis=1)
    preX_data = X_pre_resps_musc.loc[:, -0.5:-0.01].mean(axis=1)

    fig_c_pupil, (ax_cp_top, ax_cp_bot) = plt.subplots(2, 1, figsize=(2.2, 3.5), sharex=True)

    drug_states_musc = [
        ('none', 'Control'),
        ('muscimol', 'Muscimol')
    ]
    musc_pre_stim_plot_data = {}

    for ax, (stim_name, stim_data) in zip([ax_cp_top, ax_cp_bot],
                                          [('Pre pattern', preA_data), ('Pre X', preX_data)]):
        for ii, (drug_key, nickname) in enumerate(drug_states_musc):
            drug_sess_list = musc_by_drug_sess[nickname]

            for expected_val, pos_offset in zip([1, 0], [0, 1]):
                pos = expected_val + ii * 2

                mask = (
                        stim_data.index.get_level_values('sess').isin(drug_sess_list) &
                        (stim_data.index.get_level_values('expected') == expected_val)
                )
                subset = stim_data.loc[mask]
                sess_means = subset.groupby('sess').mean()

                ax.boxplot(sess_means.values, positions=[pos], **boxplot_kwargs)
                musc_pre_stim_plot_data[f'{nickname}_{stim_name}_{"expected" if expected_val == 1 else "unexpected"}'] = sess_means.values

        ax.set_ylabel('Pupil size (a.u)')
        ax.set_title(stim_name, fontsize=9)
        format_axis(ax)
        ax.set_ylim(-1.25, 1.25)

    ax_cp_bot.set_xticks([0, 1, 2, 3])
    ax_cp_bot.set_xticklabels(['Frequent', 'Rare', 'Frequent', 'Rare'])

    ax_cp_bot.text(0.5, -3.2, 'Control', ha='center', va='top', transform=ax_cp_bot.get_xaxis_transform(),
                   fontsize=8)
    ax_cp_bot.text(2.5, -3.2, 'Muscimol', ha='center', va='top', transform=ax_cp_bot.get_xaxis_transform(),
                   fontsize=8)

    fig_c_pupil.suptitle('Baseline pupil', fontsize=10, y=0.98)
    fig_c_pupil.set_layout_engine('constrained')
    fig_c_pupil.show()
    fig_c_pupil.savefig('fig_c_baseline_pupil.pdf')

    (pd.DataFrame({k:v for k,v in musc_pre_stim_plot_data.items() if 'Muscimol' in k}).T).to_csv('fig2_c_musc_pre_stim_plot_data.csv')
    (pd.DataFrame({k:v for k,v in musc_pre_stim_plot_data.items() if 'Control' in k}).T).to_csv('fig2_c_control_pre_stim_plot_data.csv')
    # =========================================================================
    # Panel g: Pre-stimulus Pupil Analysis & Boxplots (Pre pattern & Pre X)
    # =========================================================================

    # 1. Load pre-computed condition response dictionaries for Panel g (using direct path without paper_data_dir)
    A_panel_g_pre_by_cond = load_response_dfs(
        DATA_DIR / "pupil_responses" / "fig02_opto_2503_A_prestim_by_condition.h5"
    )
    X_panel_g_pre_by_cond = load_response_dfs(
        DATA_DIR / "pupil_responses" / "fig02_opto_2503_X_prestim_by_condition.h5"
    )

    A_pre_opto_dict = {}
    X_pre_opto_dict = {}

    # for cond_name in ['expected', 'unexpected']:
    for cond_name in ['normal', 'deviant_C']:
        exp_val = 1 if cond_name == 'normal' else 0
        df_a = A_panel_g_pre_by_cond[cond_name].copy()
        df_a['expected'] = exp_val
        df_a.set_index('expected', append=True, inplace=True)
        A_pre_opto_dict['expected' if cond_name =='normal' else 'unexpected'] = df_a
        df_x = X_panel_g_pre_by_cond[cond_name].copy()
        df_x['expected'] = exp_val
        df_x.set_index('expected', append=True, inplace=True)
        X_pre_opto_dict['expected' if cond_name =='normal' else 'unexpected'] = df_x

    A_pre_resps_opto = pd.concat([A_pre_opto_dict[c] for c in ['expected', 'unexpected']])
    X_pre_resps_opto = pd.concat([X_pre_opto_dict[c] for c in ['expected', 'unexpected']])

    preA_data = A_pre_resps_opto.loc[:, -0.5:-0.01].mean(axis=1)
    preX_data = X_pre_resps_opto.loc[:, -0.5:-0.01].mean(axis=1)

    fig_g_pupil, (ax_gp_top, ax_gp_bot) = plt.subplots(2, 1, figsize=(2.2, 3.5), sharex=True)

    opto_states_g = [
        ('light_off', 'Control'),
        ('light_on', 'Inhibition')
    ]
    opto_pre_stim_data = {}
    for ax, (stim_name, stim_data) in zip([ax_gp_top, ax_gp_bot], [('Pre pattern', preA_data), ('Pre X', preX_data)]):
        sess_names_old_style = pd.Series([s.rsplit('_',1)[0] for s in stim_data.index.get_level_values('sess')])
        sess_names_old_style.index = stim_data.index
        for ii, (subset_key, nickname) in enumerate(opto_states_g):
            for expected_val, pos_offset in zip([1, 0], [0, 1]):
                nd_opto_state_sess_list = opto_sess_dict[subset_key]

                pos = expected_val + ii * 2

                mask = (
                    sess_names_old_style.isin(nd_opto_state_sess_list)
                ) & (
                    stim_data.index.get_level_values('expected') == expected_val
                )
                subset = stim_data.loc[mask]
                # subset = (
                #     subset.sort_index(level=["name", "sess"])
                #     .groupby(level="name", group_keys=False)
                #     # .head(10)
                # )
                sess_means = subset.groupby('sess').mean()


                ax.boxplot(sess_means.values, positions=[pos], **boxplot_kwargs)
                opto_pre_stim_data[f'{nickname}_{stim_name}_{"expected" if expected_val == 1 else "unexpected"}'] = sess_means

        ax.set_ylabel('Pupil size (a.u)')
        ax.set_title(stim_name, fontsize=9)
        format_axis(ax)
        ax.set_ylim(-1, 1)

    ax_gp_bot.set_xticks([0, 1, 2, 3])
    ax_gp_bot.set_xticklabels(['Expected', 'Unexpected', 'Expected', 'Unexpected'])

    ax_gp_bot.text(0.5, -3.2, 'Control', ha='center', va='top', transform=ax_gp_bot.get_xaxis_transform(), fontsize=8)
    ax_gp_bot.text(2.5, -3.2, 'Inhibition', ha='center', va='top', transform=ax_gp_bot.get_xaxis_transform(), fontsize=8)

    fig_g_pupil.suptitle('g\nBaseline pupil', fontsize=10, y=0.98)
    fig_g_pupil.set_layout_engine('constrained')
    fig_g_pupil.show()

    pd.DataFrame({k:v for k,v in opto_pre_stim_data.items() if 'Inhibition' in k}).T.to_csv('fig2_g_opto_pre_stim_plot_data.csv')
    pd.DataFrame({k:v for k,v in opto_pre_stim_data.items() if 'Control' in k}).T.to_csv('fig2_g_control_pre_stim_plot_data.csv')

    # =========================================================================
    # Panel k: Pre-stimulus Pupil Analysis & Boxplots using Loaded HDF5 Dictionaries
    # =========================================================================

    # 1. Load pre-saved pre-stimulus condition response dictionaries for Panel k with unique panel-specific names
    A_panel_k_pre_by_cond = load_response_dfs(
        DATA_DIR / "pupil_responses" / "fig02_opto_deviant_A_prestim_by_condition.h5"
    )
    X_panel_k_pre_by_cond = load_response_dfs(
        DATA_DIR / "pupil_responses" / "fig02_opto_deviant_X_prestim_by_condition.h5"
    )

    dev_cond_names = ['normal', 'dev_ABCD1', 'dev_ABBA1']
    subsets_2_plot = ['control', 'opto']

    resp_dfs_by_stim_by_subset = {}

    # 2. Extract and format window means for each stimulus and subset
    for stim_name, stim_dict in zip(['A', 'X'], [A_panel_k_pre_by_cond, X_panel_k_pre_by_cond]):
        for subset_name in subsets_2_plot:
            subset_df = stim_dict[f'{subset_name}']

            resp_dfs = {}
            for cond in dev_cond_names:
                cond_df = subset_df.xs(cond, level='condition')

                resp_dfs[cond] = cond_df.loc[:, -0.5:-0.01].mean(axis=1).groupby(['sess', 'name']).mean()

            resp_dfs_by_stim_by_subset[f'{stim_name}_{subset_name}'] = resp_dfs

    # 3. Assemble DataFrames for plotting
    _pre_stim_dfs = {k: pd.DataFrame(v) for k, v in resp_dfs_by_stim_by_subset.items()}
    for k, v in _pre_stim_dfs.items():
        v.columns = [f'{k}_{col}' for col in v.columns]

    prestim_df = pd.concat([v for v in _pre_stim_dfs.values()], axis=1)
    cols_reorder = [f'{stim}_{subset}_{cond}' for stim in ['X', 'A'] for cond in dev_cond_names for subset in
                    subsets_2_plot]
    prestim_df = prestim_df[[c for c in cols_reorder if c in prestim_df.columns]]

    # 4. Boxplot generation
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

    pre_stim_boxplot = plt.subplots(nrows=2)
    for stim, ax in zip(['X', 'A'], pre_stim_boxplot[1].flatten()):
        valid_cols = [col for col in prestim_df.columns if col.split('_')[0] == stim]
        if valid_cols:
            ax.boxplot([prestim_df[col].dropna() for col in valid_cols], **boxplot_kwargs)
            format_axis(ax)
            set_nested_xlabels(
                ax, ['control', 'inhibition'] * len(dev_cond_names),
                {k: [0 + e * 2, 1 + e * 2] for e, k in enumerate(dev_cond_names)},
                x_positions=np.arange(1, len(valid_cols) + 1),
                level1_fontsize=5.5, level2_fontsize=6
            )

    pre_stim_boxplot[0].suptitle('k', fontsize=10, y=0.98)
    pre_stim_boxplot[0].set_size_inches(4, 4)
    pre_stim_boxplot[0].set_layout_engine('constrained')
    pre_stim_boxplot[0].show()
    pre_stim_boxplot[0].savefig('fig_2k_data_pre_stim_boxplot.pdf')

    prestim_df = pd.concat([v for v in _pre_stim_dfs.values()], axis=1)
    prestim_df.loc[:,[c for c in prestim_df.columns if 'control' in c]].dropna().to_csv('fig2k_rule_dev_control_by_cond_pre.csv', index=False)
    prestim_df.loc[:,[c for c in prestim_df.columns if 'control' not in c]].dropna().to_csv('fig2k_rule_dev_opto_by_cond_pre.csv', index=False)