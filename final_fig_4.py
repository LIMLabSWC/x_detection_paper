import json
import platform
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from pupil_analysis_funcs import PupilCondAnalysis, run_pupil_cond_analysis
from save_utils import save_stats_to_tex
from xdetectioncore.behaviour import get_all_cond_filts
from xdetectioncore.io_utils import extract_date, load_response_dfs
from xdetectioncore.plotting import format_axis


if __name__ == '__main__':

    plt.style.use('figure_stylesheet.mplstyle')

    # Data paths for Opto dataset
    paper_data_dir = Path(r"X:\Xdetection_data\data_for_paper")
    A_by_cond_path = paper_data_dir / "fig2_opto_A_by_cond.h5"
    all_td_df_path = paper_data_dir / "fig2_opto_all_td_df.h5"
    opto_sess_dict_path = paper_data_dir / "opto_sess_dict.json"

    # Load condition pupil response DataFrames
    A_by_cond = load_response_dfs(A_by_cond_path)

    # Load trial data DataFrame
    all_td_df = pd.read_hdf(all_td_df_path)
    assert isinstance(all_td_df, pd.DataFrame)

    # Load opto state session dictionary
    with open(opto_sess_dict_path, 'r') as f:
        opto_sess_dict = json.load(f)

    cond_filters = get_all_cond_filts()
    stats_dir = Path('stats')
    if not stats_dir.is_dir():
        stats_dir.mkdir(parents=True)

    opto_batch = [f'RS0{i}' for i in '123678']
    # opto_batch = [f'RS0{i}' for i in '2378']
    early_late_opto_states = ['light_off', 'first_10_trial_freq_block_light_on']

    # Define early/late frequent block conditions
    rare_freq_early_late_conds = ['rare'] + [f'frequent_prate_{ti}' for ti in np.arange(0, 25, 5)][:5]

    cmap = plt.get_cmap('Blues_r')
    colors = ['k'] + [cmap(i / len(rare_freq_early_late_conds)) for i in range(len(rare_freq_early_late_conds))]
    rare_freq_early_late_line_kwargs = {
        cond: {'c': col, 'ls': '-'} for cond, col in zip(rare_freq_early_late_conds, colors)
    }

    all_rare_freq_early_late_sess = [
        sess for sess in A_by_cond['rare'].index.get_level_values('sess').unique()
        if all(cond in A_by_cond and sess in A_by_cond[cond].index.get_level_values('sess') for cond in rare_freq_early_late_conds)
    ]

    boxplot_kwargs = dict(
        widths=0.4,
        patch_artist=False,
        showmeans=False,
        showfliers=False,
        medianprops=dict(mfc='k', color='maroon', lw=1.5),
        boxprops=dict(lw=0.5),
        whiskerprops=dict(lw=0.5),
        capprops=dict(lw=0.5),
    )

    # =========================================================================
    # Panel e: Time-Series Progression for All Non-Inhibition Sessions
    # =========================================================================
    non_opto_early_late_sess = [
        sess for sess in all_rare_freq_early_late_sess
        if all(sess not in opto_sess_dict.get(state, []) for state in ['light_on', 'first_10_trial_freq_block_light_on'])
    ]

    if non_opto_early_late_sess:
        analysis_all = run_pupil_cond_analysis(
            by_cond_dict=A_by_cond,
            sess_list=non_opto_early_late_sess,
            conditions=rare_freq_early_late_conds,
            figdir=Path('.'),
            line_kwargs=rare_freq_early_late_line_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            group_name='sess',
            permutation_test=False,
            n_permutations=None,
            cluster_groupname=['sess'],
            cluster_comps=[['frequent_prate_20', 'frequent_prate_0'], ['frequent_prate_20', 'frequent_prate_10']],
            stats_dir=stats_dir,
            fig_savename='panel_e_all_sessions_ts.pdf',
            tex_name='panel_e_all_sessions_stats.tex',
            ylabel="pupil size (au)",
            figsize=(1.6, 2),
            inset_max_diff=False
        )

        fig_e = analysis_all.plots['ts_by_cond'][0]
        ax_e = analysis_all.plots['ts_by_cond'][1]
        ax_e.set_xlim([-0.5, 2.0])
        ax_e.set_ylim([-0.1, 0.4])
        ax_e.set_title("All sessions")
        fig_e.set_size_inches(2.2, 2.2)
        fig_e.show()

        # Extract diff across windows for no_opto
        pupil_by_cond = [
            analysis_all.by_cond[cond].subtract(analysis_all.by_cond[cond].loc[:, -1:0].mean(axis=1), axis=0)
            for cond in ['rare'] + rare_freq_early_late_conds[1:]
        ]
        pupil_by_cond = [
            cond_resps.loc[:, 1.5:2].mean(axis=1).groupby(['sess', 'name']).median()
            for cond_resps in pupil_by_cond
        ]
        diff_across_windows_by_state = {
            'no_opto': [(pupil_by_cond[i] - pupil_by_cond[0]).groupby('name').mean() for i in range(len(pupil_by_cond))]
        }
    else:
        diff_across_windows_by_state = {}

    # =========================================================================
    # Panel f: Time-Series Progression during Frequent Block Inhibition
    # =========================================================================
    early_late_diff_by_state = {}

    for opto_state in early_late_opto_states:
        opto_state_sesslist = [
            sess for sess in all_rare_freq_early_late_sess
            if sess in opto_sess_dict.get(opto_state, []) and sess.split('_')[0] in opto_batch
            and (int(extract_date(sess)) > 250920 if opto_state != 'light_off' else 1)
        ]

        if not opto_state_sesslist:
            continue

        selected_conds = ['rare'] + rare_freq_early_late_conds[(3 if 'light_off' not in opto_state else 0):]

        analysis_opto = run_pupil_cond_analysis(
            by_cond_dict=A_by_cond,
            sess_list=opto_state_sesslist,
            conditions=selected_conds,
            figdir=Path('.'),
            line_kwargs=rare_freq_early_late_line_kwargs,
            boxplot_kwargs=boxplot_kwargs,
            window_by_stim=(1, 2),
            smoothing_window=5,
            max_window_kwargs={'max_func': 'max'},
            group_name='sess',
            permutation_test=False,
            n_permutations=None,
            cluster_groupname=['sess'],
            stats_dir=stats_dir,
            fig_savename=f'panel_f_{opto_state}_ts.pdf',
            tex_name=f'panel_f_{opto_state}_stats.tex',
            ylabel="pupil size (au)",
            figsize=(1.6, 2),
            ylim_ts=(-0.15, 0.45),
            inset_max_diff=False
        )

        fig_f = analysis_opto.plots['ts_by_cond'][0]
        ax_f = analysis_opto.plots['ts_by_cond'][1]
        ax_f.set_xlim([-0.5, 2.0])
        ax_f.set_title("Frequent block inhibition" if opto_state != 'light_off' else "Control")
        fig_f.set_size_inches(2.2, 2.2)
        fig_f.show()

        # Extract peak responses for early vs late differences
        if all(c in analysis_opto.by_cond for c in ['frequent_prate_20', 'frequent_prate_10']):
            pupil_by_cond = [
                analysis_opto.by_cond[cond].loc[:, 1.8:2.].mean(axis=1).groupby(['sess', 'name']).median()
                for cond in ['frequent_prate_20', 'frequent_prate_10']
            ]
            early_late_diff = (pupil_by_cond[-1] - pupil_by_cond[0]).groupby('name').mean()
            early_late_diff_by_state[f'{opto_state}_all_batch'] = early_late_diff

        # Baseline-subtracted window differences for adaptation timecourse
        pupil_by_cond = [
            analysis_opto.by_cond[cond].subtract(analysis_opto.by_cond[cond].loc[:, -1:0].mean(axis=1), axis=0)
            for cond in analysis_opto.by_cond.keys()
        ]
        pupil_by_cond = [
            cond_resps.loc[:, 1.8:2.].mean(axis=1).groupby(['sess', 'name']).median()
            for cond_resps in pupil_by_cond
        ]
        diff_across_windows_by_state.update({
            f'{opto_state}_all_batch': [(pupil_by_cond[i] - pupil_by_cond[0]).groupby('name').mean() for i in range(len(pupil_by_cond))]
        })

    # =========================================================================
    # Panel g (Left): Plot Diff Windows (Exact original section)
    # =========================================================================
    plt_cols = ['k', '#0066ffe6']
    diff_across_windows_plot = plt.subplots()
    ax = diff_across_windows_plot[1]
    diffs_as_dfs = [pd.concat(e, axis=1).loc[:, 1:] for ei, e in enumerate(diff_across_windows_by_state.values())]
    [
        ax.errorbar(
            _df.columns + ii * 0.05 + (2 if ii > 0 else 0),
            _df.mean(axis=0),
            yerr=_df.sem(axis=0),
            fmt='.',
            lw=2,
            alpha=0.9,
            markersize=15,
            c=plt_cols[ii]
        )
        for ii, _df in enumerate(diffs_as_dfs[1:])
    ]
    ax.axhline(0, color='k', ls='--')
    ax.set_xlim(0.75, 4.25)
    ax.set_xticks(np.arange(1, 5))
    ax.set_xticklabels([f'Freq {t} -{t+4}' for t in np.arange(1, 21, 5)])
    diff_across_windows_plot[0].set_size_inches(1, 1.5)
    diff_across_windows_plot[0].show()

    if len(diffs_as_dfs) > 2:
        print(ttest_rel(diffs_as_dfs[2][1].values, diffs_as_dfs[1][3].values, alternative='greater'))
        print(ttest_rel(diffs_as_dfs[2][1].values, diffs_as_dfs[1][3].values, alternative='two-sided'))

    # =========================================================================
    # Panel g (Right): Mid-Late Pupil Adaptation (Control vs Inhibition Boxplot)
    # =========================================================================

    if len(early_late_diff_by_state) >= 2:
        ctrl_vals = early_late_diff_by_state['light_off_all_batch'].values
        inh_vals = early_late_diff_by_state['first_10_trial_freq_block_light_on_all_batch'].values

        ttests_early_late = ttest_rel(inh_vals,ctrl_vals, alternative='greater')
        save_stats_to_tex(ttests_early_late, stats_dir / 'rarefreq_early_vs_late_no_rare_inh.tex')

        fig_g_right, ax_g_right = plt.subplots(figsize=(1.2, 2.2))
        ax_g_right.boxplot([ctrl_vals, inh_vals], labels=['no inh', 'inh'], **boxplot_kwargs)

        for i, data in enumerate([ctrl_vals, inh_vals]):
            ax_g_right.scatter([i + 1] * len(data), data, fc='k', alpha=0.5, s=12, ec='none')

        all_vals = np.hstack([ctrl_vals, inh_vals])
        y_max = np.percentile(all_vals, 97.5) + 0.05
        h = 0.02
        ax_g_right.plot([1, 1, 2, 2], [y_max, y_max + h, y_max + h, y_max], lw=0.8, c='k')
        ax_g_right.text(1.5, y_max + h + 0.01, "*" if ttests_early_late.pvalue < 0.05 else "ns",
                        ha='center', va='bottom',fontsize=6)

        format_axis(ax_g_right)
        ax_g_right.set_ylabel('Max pupil diff (frequent - rare)')
        fig_g_right.set_layout_engine('constrained')
        fig_g_right.show()