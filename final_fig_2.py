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


if __name__ == '__main__':

    plt.style.use('figure_stylesheet.mplstyle')

    # Data paths for Opto dataset
    paper_data_dir = Path(r"X:\Xdetection_data\data_for_paper")

    ###################################################################################################################
    ##### MUSCIMOL ################################################################################################
    ###################################################################################################################
    # Data paths for Muscimol dataset
    A_by_cond_path = paper_data_dir / "fig2_musc_A_by_cond.h5"
    X_by_cond_path = paper_data_dir / "fig2_musc_X_by_cond.h5"
    all_td_df_path = paper_data_dir / "fig2_musc_all_td_df.h5"
    musc_sess_dict_path = paper_data_dir / "musc_sess_dict.json"

    # Load condition pupil response DataFrames
    A_by_cond = load_response_dfs(A_by_cond_path)
    X_by_cond = load_response_dfs(X_by_cond_path)
    for _by_cond in [A_by_cond, X_by_cond]:
        add_name_to_response_dfs(_by_cond)

    # Load trial data
    all_td_df = pd.read_hdf(all_td_df_path)
    assert isinstance(all_td_df, pd.DataFrame)

    # Load drug session dictionary
    with open(musc_sess_dict_path, 'r') as f:
        drug_sess_dict = json.load(f)

    cond_filters = get_all_cond_filts()
    good_trial_filt = 'n_since_last_Trial_Outcome <=5 '

    xlim = [-0.25, 2.25]

    # Filter out excluded dates and DO76 from drug sessions
    exclude_dates = ['240710', '240712', '240716', '240717', '240726']

    none_sess = [
        sess for sess in drug_sess_dict.get('none', [])
        if sess.split('_')[1] not in exclude_dates and 'DO76' not in sess
           and 'DO71' not in sess
    ]
    musc_sess = [
        sess for sess in drug_sess_dict.get('muscimol', [])
        if sess.split('_')[1] not in exclude_dates and 'DO76' not in sess
           and 'DO71' not in sess
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
    rare_freq_line_kwargs = {
        'rare': {'c': '#00cbfcff', 'ls': '-'},
        'frequent': {'c': '#1530e9ff', 'ls': '-'}
    }

    all_rare_freq_sess = [
        sess for sess in A_by_cond['rare'].index.get_level_values('sess').unique()
        if sess in A_by_cond['frequent'].index.get_level_values('sess')
    ]

    rare_freq_by_drug_by_cond = {}

    for drug_key, drug_label in drugs_to_plot:
        drug_sess_list = sess_2_use_by_drug[drug_key]

        # Extract peak PDR in window [0.5, 2.5]s after smoothing
        resps_by_cond = {
            cond: A_by_cond[cond].T.rolling(window=25).mean().T
                  .loc[A_by_cond[cond].index.get_level_values('sess').isin(drug_sess_list), 0.5:2.5]
            .max(axis=1).groupby(['name']).mean()
            for cond in rare_freq_condnames
        }

        cond_df = pd.DataFrame.from_dict(resps_by_cond).dropna()
        rare_freq_by_drug_by_cond[drug_key] = cond_df

        # --- Figure: Peak PDR per condition for current drug state ---
        fig, ax = plt.subplots(figsize=(1.5, 2))
        ax.boxplot([cond_df[c] for c in rare_freq_condnames], labels=rare_freq_condnames, **boxplot_kwargs)

        for c_idx, cond in enumerate(rare_freq_condnames):
            vals = cond_df[cond].values
            ax.scatter([c_idx + 1] * len(vals), vals, fc=rare_freq_line_kwargs[cond]['c'], alpha=0.6, s=15,
                       ec='none')

        for _, row in cond_df.iterrows():
            ax.plot([1, 2], [row['frequent'], row['rare']], lw=0.3, c='gray', alpha=0.7)

        format_axis(ax)
        ax.set_ylabel('Peak PDR (a.u.)')
        ax.set_title(f'Peak PDR - {drug_label}')
        fig.set_layout_engine('constrained')
        fig.show()

    # --- Figure: Combined Peak PDR Comparison Across Control & Muscimol ---
    fig, ax = plt.subplots(figsize=(2.2, 2))

    x_positions = [0.8, 1.2, 2.8, 3.2]
    plot_data = []

    for drug_key in ['none', 'muscimol']:
        for cond in rare_freq_condnames:
            plot_data.append(rare_freq_by_drug_by_cond[drug_key][cond].values)

    ax.boxplot(plot_data, positions=x_positions, **boxplot_kwargs)

    # Scatter points & connect paired animals within each drug condition
    for d_idx, (drug_key, _) in enumerate(drugs_to_plot):
        df = rare_freq_by_drug_by_cond[drug_key]
        pos_pair = [x_positions[d_idx * 2], x_positions[d_idx * 2 + 1]]

        for cond_idx, cond in enumerate(rare_freq_condnames):
            ax.scatter(
                [pos_pair[cond_idx]] * len(df),
                df[cond].values,
                fc=rare_freq_line_kwargs[cond]['c'],
                alpha=0.6,
                s=12,
                ec='none'
            )

        for _, row in df.iterrows():
            ax.plot(pos_pair, [row['frequent'], row['rare']], lw=0.3, c='gray', alpha=0.7)

    format_axis(ax)
    ax.set_xticks([1.0, 3.0])
    ax.set_xticklabels(['Control', 'Muscimol'])
    ax.set_ylabel('Peak PDR (a.u.)')
    fig.set_layout_engine('constrained')
    fig.show()

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

    # =========================================================================
    # 2. X Response Plot (hit_all vs miss_all)
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

    # =========================================================================
    # 3. Rare vs Frequent Time-Series Plot (Control vs Muscimol)
    # =========================================================================
    for stim_name, stim_by_cond in zip(['pattern', 'X'], [A_by_cond, X_by_cond]):
        for drug_key, drug_label in drugs_to_plot:
            target_sess = sess_2_use_by_drug[drug_key]
            if not target_sess:
                continue
            rf_analysis = PupilCondAnalysis(stim_by_cond, rare_freq_condnames, target_sess,
                                            event_name=stim_name)
            rf_analysis.plot_ts_by_cond(cond_line_kwargs=rare_freq_line_kwargs)

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

    # =========================================================================
    # 4. Behavioral Performance Summary Across Tone Positions
    # =========================================================================
    patt_nonpatt_cols = ['dimgray', 'indigo']

    for drug_key, drug_label in drugs_to_plot:
        drug_sess_list = [sess for sess in drug_sess_dict.get(drug_key, []) if 'DO76' not in sess]
        rare_freq_sess_df = all_td_df.loc[all_td_df.index.get_level_values('sess').isin(drug_sess_list)]

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
        # for a_data in perf_data:
        #     ax.plot(['Non pattern', 'Pattern'], a_data, lw=0.25, color='gray')

        ax.set_ylim(0.4, 1.01)
        format_axis(ax)
        ax.set_ylabel('Hit rate')
        ax.set_title(f'{drug_label}')
        fig.set_layout_engine('constrained')
        fig.show()


    # =========================================================================
    # 5. Overall Performance & Reaction Time Summary Across Tone Positions
    # =========================================================================
    patt_nonpatt_cols = ['dimgray', 'indigo']

    for drug_key, drug_label in drugs_to_plot:
        drug_sess_list = [sess for sess in drug_sess_dict.get(drug_key, []) if 'DO76' not in sess]
        rare_freq_sess_df = all_td_df.loc[all_td_df.index.get_level_values('sess').isin(drug_sess_list)]

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
    # Opto (Normal vs Deviant ABAD
    # =========================================================================

    A_by_cond_path = paper_data_dir / "fig2_opto_A_by_cond.h5"
    X_by_cond_path = paper_data_dir / "fig2_opto_X_by_cond.h5"
    all_td_df_path = paper_data_dir / "fig2_opto_all_td_df.h5"
    opto_sess_dict_path = paper_data_dir / "opto_sess_dict.json"

    # Load condition pupil response DataFrames
    A_by_cond = load_response_dfs(A_by_cond_path)
    X_by_cond = load_response_dfs(X_by_cond_path)

    # Load trial data
    all_td_df = pd.read_hdf(all_td_df_path)
    assert isinstance(all_td_df, pd.DataFrame)

    # Load opto state session dictionary
    with open(opto_sess_dict_path, 'r') as f:
        opto_sess_dict = json.load(f)

    cond_filters = get_all_cond_filts()
    good_trial_filt = 'n_since_last_Trial_Outcome <=5 '

    xlim = [-0.25, 2.25]

    # Mouse cohorts
    opto_batch = [f'RS0{i}' for i in '123678']
    control_batch = [f'RS0{i}' for i in '459']

    # Identify block/pattern sessions
    abcd_sess = all_td_df.query(
        '(PatternID == "10;15;20;25" or PatternID=="10;12;14;16") and (Session_Block==0 or Session_Block==2)'
    ).index.get_level_values('sess').unique()

    abad_sess = all_td_df.query(
        'PatternID == "10;15;10;25" and (Session_Block==0 or Session_Block==2)'
    ).index.get_level_values('sess').unique()

    #
    # # =========================================================================
    # # 2. Rare vs Frequent Analysis Across Opto States
    # # =========================================================================
    # rare_freq_line_kwargs = {
    #     'rare': {'c': '#00cbfcff', 'ls': '-'},
    #     'frequent': {'c': '#1530e9ff', 'ls': '-'}
    # }
    # rare_freq_condnames = ['frequent', 'rare']
    #
    # all_rare_freq_sess = [
    #     sess for sess in A_by_cond['rare'].index.get_level_values('sess').unique()
    #     if sess in A_by_cond['frequent'].index.get_level_values('sess')
    # ]
    #
    # # Time-series analysis for Light Off state
    # opto_state_sesslist = [
    #     sess for sess in all_rare_freq_sess
    #     if sess in opto_sess_dict.get('light_off', [])
    #        and sess.split('_')[0] in opto_batch
    #        and sess in abcd_sess
    # ]
    #
    # if opto_state_sesslist:
    #     rf_analysis = PupilCondAnalysis(A_by_cond, rare_freq_condnames, opto_state_sesslist)
    #     rf_analysis.plot_ts_by_cond(cond_line_kwargs=rare_freq_line_kwargs)
    #     rf_analysis.plots['ts_by_cond'][1].set_xlim(xlim)
    #     rf_analysis.plots['ts_by_cond'][1].set_title('Rare vs Freq (Light Off)')
    #     rf_analysis.plots['ts_by_cond'][0].set_size_inches(2, 2)
    #     rf_analysis.plots['ts_by_cond'][0].show()
    #
    # # Behavioral performance: Rare vs Frequent
    # cond_td_df = all_td_df.loc[all_td_df.index.get_level_values('sess').isin(all_rare_freq_sess)]
    # rare_freq_perf = [
    #     cond_td_df.query(good_trial_filt).query(cond_filters[cond])['Trial_Outcome'].groupby('name').mean()
    #     for cond in rare_freq_condnames
    #     if cond in cond_filters
    # ]
    #
    # if rare_freq_perf:
    #     fig, ax = plt.subplots()
    #     ax.bar(rare_freq_condnames, [e.mean() for e in rare_freq_perf], ec='black', fc='w', lw=0.5, width=0.4)
    #     for cond, data in zip(rare_freq_condnames, rare_freq_perf):
    #         ax.scatter([cond] * len(data), data, edgecolor='none', fc=rare_freq_line_kwargs[cond]['c'], alpha=0.5)
    #
    #     format_axis(ax)
    #     fig.set_size_inches(1.5, 2)
    #     fig.set_layout_engine('constrained')
    #     fig.show()

    # =========================================================================
    # 3. Normal vs Deviant Analysis Across Opto States
    # =========================================================================
    norm_dev_line_kwargs = {
        'normal': {'c': 'black', 'ls': '-'},
        'deviant_C': {'c': '#cd2727ff', 'ls': '-'}
    }
    norm_dev_cond_names = ['normal', 'deviant_C']

    norm_dev_sess = [
        sess for sess in A_by_cond['normal'].index.get_level_values('sess').unique()
        if 'deviant_C' in A_by_cond and sess in A_by_cond['deviant_C'].index.get_level_values('sess')
    ]

    # Time-series analysis across conditions & stimuli
    for stim_name, stim_by_cond in zip(['A', 'X'], [A_by_cond, X_by_cond]):
        for opto_state in ['light_off', 'learning_block_light_on']:
            target_sess = [
                sess for sess in norm_dev_sess
                if sess in opto_sess_dict.get(opto_state, [])
                   and sess.split('_')[0] in opto_batch
            ]
            if target_sess:
                nd_analysis = PupilCondAnalysis(stim_by_cond, norm_dev_cond_names, target_sess)
                nd_analysis.plot_ts_by_cond(cond_line_kwargs=norm_dev_line_kwargs)
                nd_analysis.plots['ts_by_cond'][1].set_xlim(xlim)
                nd_analysis.plots['ts_by_cond'][1].set_title(f'{stim_name} Norm/Dev ({opto_state})')
                nd_analysis.plots['ts_by_cond'][0].set_size_inches(2, 2)
                nd_analysis.plots['ts_by_cond'][0].show()

    # Behavioral performance: Normal vs Deviant
    cond_td_df = all_td_df.loc[all_td_df.index.get_level_values('sess').isin(norm_dev_sess)]
    norm_dev_perf = [
        cond_td_df.query(good_trial_filt).query(cond_filters[cond])['Trial_Outcome'].groupby('name').mean()
        for cond in norm_dev_cond_names
        if cond in cond_filters
    ]

    if norm_dev_perf:
        fig, ax = plt.subplots()
        ax.bar(norm_dev_cond_names, [e.mean() for e in norm_dev_perf], ec='black', fc='w', lw=0.5, width=0.4)
        for cond, data in zip(norm_dev_cond_names, norm_dev_perf):
            ax.scatter([cond] * len(data), data, edgecolor='none', fc=norm_dev_line_kwargs[cond]['c'], alpha=0.5)

        format_axis(ax)
        fig.set_size_inches(1.5, 2)
        fig.set_layout_engine('constrained')
        fig.show()

    # =========================================================================
    # Opto Normal vs rule violating/conforming deviants
    # =========================================================================

    # Path to pre-saved HDF5 file containing all subset responses
    subset_resps_path = Path(r'D:\paper_data\fig2\gen_opto_all_subset_resps.h5')

    all_subset_resps = load_response_dfs(subset_resps_path)

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

        analysis_obj = PupilCondAnalysis(by_cond_dict=by_cond,conditions=dev_cond_names)
        analysis_obj.plot_ts_by_cond(cond_line_kwargs=dev_line_kwargs,)
        analysis_obj.plots['ts_by_cond'][1].set_title(subset_name)
        analysis_obj.plots['ts_by_cond'][0].show()

    # --- Difference Across Windows Analysis ---
    for subset_name, subset_df in all_subset_resps.items():
        assert isinstance(subset_df, pd.DataFrame)
        # Reconstruct dict of condition DataFrames
        by_cond = {
            cond: subset_df.xs(cond, level='condition')
            for cond in dev_cond_names
            if cond in subset_df.index.get_level_values('condition')
        }

        # Baseline subtract and average window (0.7s to 1.25s)
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

        # Plot rule differences across windows
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
    dev_diff_by_subset_df = {
        k: pd.DataFrame.from_dict(v, orient='columns')
        for k, v in mean_pupil_by_subset.items()
    }

    if 'opto' in dev_diff_by_subset_df and 'post_opto' in dev_diff_by_subset_df:
        for cond in dev_cond_names[1:]:
            opto_vals = dev_diff_by_subset_df['opto'][cond].groupby('name').mean()
            post_vals = dev_diff_by_subset_df['post_opto'][cond].groupby('name').mean()
            print(f"{cond}: {ttest_rel(opto_vals, post_vals)}")


    # # --- Pre-stimulus Pupil Size Boxplot Analysis ---
    # resp_dfs_by_stim_by_subset = {}
    # for subset_name, subset_df in all_subset_resps.items():
    #     resp_dfs = {}
    #     for cond in dev_cond_names:
    #         if cond in subset_df.index.get_level_values('condition'):
    #             cond_df = subset_df.xs(cond, level='condition')
    #             resp_dfs[cond] = cond_df.loc[:, -0.5:-0.25].mean(axis=1).groupby(['sess', 'name']).mean()
    #     resp_dfs_by_stim_by_subset[f'A_{subset_name}'] = resp_dfs
    #
    # subsets_2_plot = ['post_opto', 'opto']
    # _pre_stim_dfs = {
    #     k: pd.DataFrame(v)
    #     for k, v in resp_dfs_by_stim_by_subset.items()
    #     if '_'.join(k.split('_')[1:]) in subsets_2_plot
    # }
    #
    # for k, v in _pre_stim_dfs.items():
    #     v.columns = [f'{k}_{col}' for col in v.columns]
    #
    # if _pre_stim_dfs:
    #     prestim_df = pd.concat(list(_pre_stim_dfs.values()), axis=1)
    #     cols_reorder = [
    #         f'A_{subset}_{cond}'
    #         for cond in dev_cond_names
    #         for subset in subsets_2_plot
    #         if f'A_{subset}_{cond}' in prestim_df.columns
    #     ]
    #     prestim_df = prestim_df[cols_reorder]
    #
    #     boxplot_kwargs = dict(
    #         widths=0.5,
    #         patch_artist=False,
    #         showmeans=False,
    #         showfliers=False,
    #         medianprops=dict(mfc='k'),
    #         boxprops=dict(lw=0.5),
    #         whiskerprops=dict(lw=0.5),
    #         capprops=dict(lw=0.5),
    #     )
    #
    #     fig, axes = plt.subplots(nrows=2)
    #     axes_flat = np.array(axes).flatten()
    #
    #     for stim, ax in zip(['A'], axes_flat):
    #         sub_cols = [col for col in prestim_df.columns if col.split('_')[0] == stim]
    #         if sub_cols:
    #             ax.boxplot([prestim_df[col].dropna() for col in sub_cols], **boxplot_kwargs)
    #             format_axis(ax)
    #             set_nested_xlabels(
    #                 ax,
    #                 ['control', 'inhibition'] * 3,
    #                 {k: [0 + e * 2, 1 + e * 2] for e, k in enumerate(dev_cond_names)},
    #                 x_positions=np.arange(1, 7),
    #                 level1_fontsize=5.5,
    #                 level2_fontsize=6,
    #             )
    #
    #     fig.set_size_inches(4, 4)
    #     fig.show()
    #     fig.savefig('pre_stim_boxplot.pdf')

    # =========================================================================
    # Paired / One-Sample t-test on Rule Violation Difference Scores
    # d_i = (violating - conforming)_control - (violating - conforming)_inhibition
    # =========================================================================

    assert 'control' in dev_diff_by_subset_df and 'opto' in dev_diff_by_subset_df

    # Calculate violating - conforming difference for Control (post_opto) and Inhibition (opto)
    diff_control = (
            dev_diff_by_subset_df['control']['dev_ABBA1']
            - dev_diff_by_subset_df['control']['dev_ABCD1']
    ).groupby('name').mean()
    diff_inhibition = (
            dev_diff_by_subset_df['opto']['dev_ABBA1']
            - dev_diff_by_subset_df['opto']['dev_ABCD1']
    ).groupby('name').mean()

    # Align mouse names across both subsets
    common_mice = diff_control.index.intersection(diff_inhibition.index)
    diff_control = diff_control.loc[common_mice]
    diff_inhibition = diff_inhibition.loc[common_mice]

    # Calculate difference of differences per mouse d_i
    d_i = diff_control - diff_inhibition

    # One-sample t-test against 0 (df = N - 1 = 3)
    t_stat, p_val = ttest_1samp(d_i, popmean=0)

    print("\n--- One-Sample t-test on Rule Violation Difference Scores (d_i) ---")
    print(f"Mice included (N={len(d_i)}): {list(d_i.index)}")
    print(f"d_i per mouse:\n{d_i}")
    print(f"Mean d_i: {d_i.mean():.4f} +/- {d_i.sem():.4f} (SEM)")
    print(f"t({len(d_i) - 1}) = {t_stat:.4f}, p = {p_val:.4f}")

    dev_diff_by_subset = {}
    mean_pupil_by_subset = {}
