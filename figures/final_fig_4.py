import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from cycler import cycler
from copy import deepcopy as copy

from scipy.spatial import procrustes
from scipy.stats import ttest_rel, kendalltau

from analysis_scripts.pupil_analysis_funcs import run_pupil_cond_analysis
from analysis_scripts.save_utils import save_stats_to_tex
from xdetectioncore.behaviour import get_all_cond_filts
from xdetectioncore.io_utils import extract_date, load_response_dfs
from xdetectioncore.plotting import format_axis
from analysis_scripts.aggregate_psth_analysis import AggregateSession

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


if __name__ == '__main__':

    plt.style.use('../figure_stylesheet.mplstyle')
    colors = ['k']+['#800000ff','#d40000ff','#2929d6ff','#6099f2ff','#b3f3ffff']
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

    # =========================================================================
    # Panel e: Time-Series Progression for All Non-Inhibition Sessions
    # =========================================================================

    aggr_savedir = DATA_DIR / "aggregate_cache"
    pkl_dir = aggr_savedir


    # Dummy args structure expected by AggregateSession
    class DummyArgs:
        event_responses_pkl = str(aggr_savedir / "dummy.pkl")


    args = DummyArgs()

    plot_config_path = (
        Path(__file__).resolve().parents[1]
        / "analysis_configs"
        / "rare_freq_plot_configs.yaml"
    )

    assert plot_config_path.is_file()
    with open(plot_config_path, 'r') as f:
        plot_config = yaml.safe_load(f)

    conds = ['rare']+ [f'frequent_prate_{ti}' for ti in np.arange(0,30,5)][:5]

    rare_freq_resps = AggregateSession(None, args, plot_config, ['A-0'] * len(conds))
    rare_freq_resps.aggregate_mean_sess_responses(conds=conds,
                                                  concat_savename=aggr_savedir / 'fig04_rare_frequency_response_aggregate.joblib',
                                                  reload_save=True,
                                                  zscore_flag=True)
    rare_freq_resps.pca_pseudo_pop(pca_name='rare_freq_early_late_pca', standardise=False)

    # --- BY-ANIMAL PCA IMPLEMENTATION ---
    pca_name = 'rare_freq_early_late_pca_by_animal'

    # Extract unique animal IDs from the concatenated session names
    all_animals = np.unique([sess.split('_')[0] for sess in rare_freq_resps.concatenated_sessnames])

    # 1. Run PCA for each animal individually
    old_event_names = list(rare_freq_resps.concatenated_event_responses.keys())
    old_event_names[0] = 'A-0_rare_all_rare'
    rare_freq_resps.concatenated_event_responses = {k: v for k, v in zip(old_event_names,
                                                                         rare_freq_resps.concatenated_event_responses.values())}
    for animal in all_animals:
        rare_freq_resps.pca_pseudo_pop(pca_name=pca_name, by_animal=True, animal=animal, standardise=True)

    # 2. Consolidate projections across all animals
    # Use the last processed animal's PopPCA object as the base template
    rare_freq_resps.pca[pca_name] = copy(rare_freq_resps.pca[f'{pca_name}_{all_animals[-1]}'])

    proj_by_name = {}
    for animal in all_animals:
        # Access the projections for this specific animal
        animal_proj = copy(rare_freq_resps.pca[f'{pca_name}_{animal}'].projected_pca_ts_by_cond['by_class'])

        # Append the animal ID to the condition keys (e.g., 'rare_DO81')
        animal_proj = {f'{pip}_{animal}': animal_proj[pip] for pip in animal_proj}

        # Add to the consolidated dictionary
        for k, v in animal_proj.items():
            proj_by_name[k] = v

    # Assign the consolidated dictionary back to the base PCA object
    rare_freq_resps.pca[pca_name].projected_pca_ts_by_cond = {'by_class': proj_by_name}

    # 3. Plotting the consolidated PCA
    plot_cfg = plot_config.get(pca_name, {})
    plot_cfg['plot_kwargs']['ref_id'] = 3
    plot_cfg['plot_kwargs']['smoothing'] = 25
    plot_cfg['plot_kwargs']['t_end'] = 1.0
    plot_cfg['plot_kwargs']['plot_out_event'] = False
    plot_cfg['plot_kwargs']['align_trajs'] = False
    plot_cfg['plot_kwargs']['align_window'] = [-0, 1.25]
    plot_cfg['fig_kwargs']['figsize'] = [6, 4]

    for pca_combs in [[0,3]]:
        rare_freq_resps.plot_2d_pca(f'{pca_name}_DO97', pca_combs[::-1], Path(''), plot_cfg)

    # =========================================================================
    # QUANTIFICATION OF PC3 TRAJECTORY MONOTONICITY (WHOLE-TRAJECTORY ALIGNMENT)
    # =========================================================================

    x_ser = rare_freq_resps.x_ser
    late_trial_win_s = [0.9, 1.1]
    LATE_TRIAL_win = [np.where(x_ser == t)[0][0] for t in late_trial_win_s]
    align_win_s = [0, 1.25]
    align_win = [np.where(x_ser == t)[0][0] for t in align_win_s]

    MAX_TRANSACTION_BLOCK = 4
    # Sub-select features for structural space mapping
    # pc_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    pc_indices = np.arange(13).tolist()
    target_pc_idx = 3  # Index corresponding to PC3 inside the pc_indices subset array
    ref_idx = 3

    conds = \
    rare_freq_resps.pca[f'rare_freq_early_late_pca_by_animal_{list(all_animals)[0]}'].projected_pca_ts_by_cond[
        'by_class'].keys()

    # for target_pc_idx in [0,1,2,3,4]:
    for target_pc_idx in [1, 3]:
        pca_name = 'rare_freq_early_late_pca_by_animal'
        by_class_dict = rare_freq_resps.pca[pca_name].projected_pca_ts_by_cond['by_class']

        # 1. Build continuous trajectories by stacking all time bins chronologically across blocks
        animal_matrices = {}
        time_bins_per_block = None

        for animal in all_animals:
            stacked_blocks = []
            for cond in conds:
                proj = by_class_dict[f'{cond}_{animal}']  # Shape: (n_pcs, n_time_bins)
                proj_selected = proj[pc_indices, :]  # Shape: (3, n_time_bins)

                if time_bins_per_block is None:
                    time_bins_per_block = proj.shape[1]

                stacked_blocks.append(proj_selected.T)  # Shape: (n_time_bins, 3)

            # Concatenate matrices vertically to capture whole-session spatiotemporal dynamics
            animal_matrices[animal] = np.vstack(stacked_blocks)  # Shape: (len(conds) * n_time_bins, 3)


        # Helper function to extract late window average from a block inside an aligned continuous matrix
        def extract_late_window_coordinate(aligned_matrix, block_idx, bins_per_block, win_limits, pc_idx):
            start_row = block_idx * bins_per_block
            end_row = (block_idx + 1) * bins_per_block
            block_data = aligned_matrix[start_row:end_row, :]
            late_window_data = block_data[win_limits[0]:win_limits[1], pc_idx]
            return np.mean(late_window_data)


        # 2. Execute rigid Procrustes registration over full continuous trajectories
        ref_animal = all_animals[ref_idx]
        ref_matrix = animal_matrices[ref_animal]
        aligned_pc_data = []

        # Process and add reference coordinates
        mtx1_standardized, _, _ = procrustes(ref_matrix, ref_matrix)
        pd.DataFrame(mtx1_standardized).to_csv('fig_3c_pca_trajs.csv')

        for block_idx, cond in enumerate(conds):
            val = extract_late_window_coordinate(mtx1_standardized, block_idx, time_bins_per_block, LATE_TRIAL_win,
                                                 target_pc_idx)
            aligned_pc_data.append({
                'Animal': ref_animal,
                'Block_Index': block_idx,
                'PC_Aligned': val
            })

        # Align all other targets onto the consensus multi-block template
        for animal in [a for a in all_animals if a != ref_animal]:
            target_matrix = animal_matrices[animal]
            _, mtx2_aligned, _ = procrustes(ref_matrix, target_matrix)
            for block_idx, cond in enumerate(conds):
                val = extract_late_window_coordinate(mtx2_aligned, block_idx, time_bins_per_block, LATE_TRIAL_win,
                                                     target_pc_idx)
                aligned_pc_data.append({
                    'Animal': animal,
                    'Block_Index': block_idx,
                    'PC_Aligned': val
                })

        df_procrustes = pd.DataFrame(aligned_pc_data)

        # 3. Compute group-level statistics (Mean and SEM across aligned coordinates)
        summary_stats = df_procrustes.groupby('Block_Index')['PC_Aligned'].agg(['mean', 'sem']).reset_index()

        # 4. Perform trend tests using the specified block threshold
        df_transition = df_procrustes[df_procrustes['Block_Index'] <= MAX_TRANSACTION_BLOCK]
        tau_trans, p_trans = kendalltau(df_transition['Block_Index'],
                                        df_procrustes.loc[df_transition.index, 'PC_Aligned'])
        tau_full, p_full = kendalltau(df_procrustes['Block_Index'], df_procrustes['PC_Aligned'])

        print(
            f"\nProcrustes Transition Trend (Rare -> Freq {MAX_TRANSACTION_BLOCK}): tau={tau_trans:.3f}, p={p_trans:.4e}")
        print(f"Procrustes Full Session Trend: tau={tau_full:.3f}, p={p_full:.4e}")

        # 5. Generate figure strictly utilizing Object-Oriented canvas structures (no plt calls)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))

        # Plot individual aligned trajectory entries
        for animal in all_animals:
            df_sub = df_procrustes[df_procrustes['Animal'] == animal]
            ax.plot(df_sub['Block_Index'], df_sub['PC_Aligned'], alpha=0.4,
                    linewidth=1, label=animal)

        # Plot population profiles
        ax.errorbar(summary_stats['Block_Index'], summary_stats['mean'], yerr=summary_stats['sem'],
                    fmt='-o', color='black', linewidth=2, capsize=4, elinewidth=1.5,
                    label='Group Mean ± SEM')

        # Formatting axes via explicit handle parameters
        ax.set_xlabel('Block Transition')
        ax.set_ylabel(f'Procrustes Aligned PC{target_pc_idx + 1} Coordinate')
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(['Rare'] + [f'Freq {i}' for i in range(1, len(conds))], rotation=30)
        ax.set_title(
            f'Procrustes Aligned Trajectory Shift\nTransition $\\tau$ = {tau_trans:.2f} ($p$ = {p_trans:.2e})',
            fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, loc='best', fontsize=8)

        fig.tight_layout()
        fig.show()
        # fig.savefig(Path('') / f'pc{target_pc_idx + 1}_procrustes_trajectory_alignment_analysis.pdf', dpi=300)

    # Data paths for Opto dataset
    A_by_cond_path = (
        DATA_DIR / "pupil_responses" / "fig02_opto_A_pupil_by_condition.h5"
    )
    all_td_df_path = DATA_DIR / "trial_tables" / "fig04_opto_trials.h5"
    opto_sess_dict_path = (
        DATA_DIR / "session_metadata" / "fig02_fig04_opto_sessions_by_state.json"
    )

    # Load condition pupil response DataFrames
    A_by_cond = load_response_dfs(A_by_cond_path)

    # Load trial data DataFrame
    all_td_df = pd.read_hdf(all_td_df_path)
    assert isinstance(all_td_df, pd.DataFrame)

    # Load opto state session dictionary
    with open(opto_sess_dict_path, 'r') as f:
        opto_sess_dict = json.load(f)

    cond_filters = get_all_cond_filts()
    stats_dir = Path('../stats')
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

    analysis_all = run_pupil_cond_analysis(
        by_cond_dict=A_by_cond,
        sess_list=non_opto_early_late_sess,
        conditions=rare_freq_early_late_conds,
        figdir=Path('..'),
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

    fig_4e_plot_means = {f'{k}_mean': v.groupby('sess').mean().mean() for k,v in analysis_all.by_cond.items()}
    fig_4e_plot_sems = {f'{k}_sem': v.groupby('sess').mean().sem() for k,v in analysis_all.by_cond.items()}
    pd.DataFrame((fig_4e_plot_sems|fig_4e_plot_sems)).to_csv('fig_4e_plot_sems.csv')

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


        selected_conds = ['rare'] + rare_freq_early_late_conds[(3 if 'light_off' not in opto_state else 0):]

        analysis_opto = run_pupil_cond_analysis(
            by_cond_dict=A_by_cond,
            sess_list=opto_state_sesslist,
            conditions=selected_conds,
            figdir=Path('..'),
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

        fig_4f_plot_means = {f'{k}_mean': v.groupby('sess').mean().mean() for k,v in analysis_opto.by_cond.items()}
        fig_4f_plot_sems = {f'{k}_sem': v.groupby('sess').mean().sem() for k,v in analysis_opto.by_cond.items()}
        pd.DataFrame((fig_4f_plot_means|fig_4f_plot_sems)).to_csv(f'fig_4f_{opto_state}_plot_data.csv')

        # Extract peak responses for early vs late differences
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

    block_names = [f'freq_{i}-{i+4}' for i in np.arange(1, 21, 5)]
    block_names = block_names+block_names[-2:]
    ctrl_vals_2_save = diffs_as_dfs[1].values[:,:4]
    inh_vals_2_save = diffs_as_dfs[2].values[:,:2]
    block_names = [f'{"control" if ii < 4 else "inhibition"}_{b}' for ii, b in enumerate(block_names)]
    pd.DataFrame(np.hstack([ctrl_vals_2_save, inh_vals_2_save]),columns=block_names).to_csv('fig_4g_left_data.csv')

    print(ttest_rel(diffs_as_dfs[2][1].values, diffs_as_dfs[1][3].values, alternative='greater'))
    print(ttest_rel(diffs_as_dfs[2][1].values, diffs_as_dfs[1][3].values, alternative='two-sided'))

    # =========================================================================
    # Panel g (Right): Mid-Late Pupil Adaptation (Control vs Inhibition Boxplot)
    # =========================================================================

    ctrl_vals = early_late_diff_by_state['light_off_all_batch'].values
    inh_vals = early_late_diff_by_state['first_10_trial_freq_block_light_on_all_batch'].values

    ttests_early_late = ttest_rel(inh_vals,ctrl_vals, alternative='greater')
    save_stats_to_tex(ttests_early_late, stats_dir / 'rarefreq_early_vs_late_no_rare_inh.tex')

    fig_g_right, ax_g_right = plt.subplots(figsize=(1.2, 2.2))
    ax_g_right.boxplot([ctrl_vals, inh_vals], labels=['no inh', 'inh'], **boxplot_kwargs)

    pd.DataFrame(np.vstack([ctrl_vals, inh_vals]), index=['no inh', 'inh']).to_csv('fig4_g_boxplot.csv')

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
    

    # =========================================================================
    # Panel h: Baseline Pupil (Pre pattern and Pre X) for Control vs Inhibition (Mid vs Late)
    # =========================================================================

    # Load the required response dictionaries using A_pre_by_cond and X_pre_by_cond
    A_pre_by_cond = load_response_dfs(
        DATA_DIR / "pupil_responses" / "fig04_opto_A_prestim_by_condition.h5"
    )
    X_pre_by_cond = load_response_dfs(
        DATA_DIR / "pupil_responses" / "fig04_opto_X_prestim_by_condition.h5"
    )

    conds_by_expected_h = {
        'unexpected': ['frequent_prate_10'],  # mid
        'expected': ['frequent_prate_20']  # late
    }

    # Tag and prepare A_pre and X_pre responses by expected status
    A_pre_h_dict = {}
    X_pre_h_dict = {}

    for cond_name, cond_list in conds_by_expected_h.items():
        exp_val = 1 if cond_name == 'expected' else 0
        for c in cond_list:
            if c in A_pre_by_cond:
                df_a = A_pre_by_cond[c].copy()
                df_a['expected'] = exp_val
                A_pre_h_dict[c] = df_a
            if c in X_pre_by_cond:
                df_x = X_pre_by_cond[c].copy()
                df_x['expected'] = exp_val
                X_pre_h_dict[c] = df_x

    A_pre_resps_h = pd.concat(list(A_pre_h_dict.values()))
    X_pre_resps_h = pd.concat(list(X_pre_h_dict.values()))

    # Extract baseline windows (Pre pattern and Pre X)
    preA_data = A_pre_resps_h.loc[:, -1:0.5].mean(
        axis=1) if -1 in A_pre_resps_h.columns or A_pre_resps_h.columns.min() < 0 else A_pre_resps_h.iloc[:, :int(
        A_pre_resps_h.shape[1] / 2)].mean(axis=1)
    preX_data = X_pre_resps_h.loc[:, -0.5:-0.01].mean(
        axis=1) if -0.5 in X_pre_resps_h.columns else X_pre_resps_h.iloc[:, :int(X_pre_resps_h.shape[1] / 3)].mean(
        axis=1)

    fig_h, (ax_h_top, ax_h_bot) = plt.subplots(2, 1, figsize=(2.2, 3.5), sharex=True)

    opto_states_h = [
        ('light_off', 'Control'),
        ('first_10_trial_freq_block_light_on', 'Inhibition')
    ]

    fig_4h_data = {}
    for ax, (stim_name, stim_data) in zip([ax_h_top, ax_h_bot], [('Pre pattern', preA_data), ('Pre X', preX_data)]):
        for ii, (opto_state, nickname) in enumerate(opto_states_h):
            state_sess_list = opto_sess_dict.get(opto_state, [])

            for expected_val, pos_offset in zip([0, 1], [0, 1]):
                pos = expected_val + ii * 2

                mask = (
                        stim_data.index.get_level_values('sess').isin(state_sess_list) &
                        stim_data.index.get_level_values('name').isin(opto_batch) &
                        (stim_data.index.get_level_values(
                            'expected') == expected_val if 'expected' in stim_data.index.names else True)
                )
                subset = stim_data.loc[mask]
                sess_means = subset.groupby('sess').mean()
                
                ax.boxplot(sess_means.values, positions=[pos], **boxplot_kwargs)
                fig_4h_data[f'{stim_name}_{nickname}_{"mid" if expected_val == 0 else "late"}'] = sess_means.values

        ax.set_ylabel('Pupil size (a.u)')
        ax.set_title(stim_name, fontsize=9)
        format_axis(ax)
        ax.set_ylim(-2.5, 2.5)

        ax_h_bot.set_xticks([0, 1, 2, 3])
        ax_h_bot.set_xticklabels(['mid', 'late', 'mid', 'late'])

        ax_h_bot.text(0.5, -3.2, 'Control', ha='center', va='top', transform=ax_h_bot.get_xaxis_transform(), fontsize=8)
        ax_h_bot.text(2.5, -3.2, 'Inhibition', ha='center', va='top', transform=ax_h_bot.get_xaxis_transform(),
                      fontsize=8)

    fig_h.suptitle('h\nBaseline pupil', fontsize=10, y=0.98)
    fig_h.set_layout_engine('constrained')
    fig_h.show()
    fig_h.savefig('panel_h_baseline_pupil.pdf', format='pdf')

    pd.DataFrame({k:v for k,v in fig_4h_data.items() if 'Control' in k}).to_csv('panel_h_baseline_pupil_control.csv')
    pd.DataFrame({k:v for k,v in fig_4h_data.items() if 'Inh' in k}).to_csv('panel_h_baseline_pupil_inh.csv')