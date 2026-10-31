from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml
from typing import List, Tuple, Union, Optional

import pingouin as pg
from scipy.stats import ttest_ind, ttest_1samp, ttest_rel, wilcoxon, kendalltau

import joblib
from cycler import cycler
from tqdm import tqdm
from scipy.stats import ttest_rel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from aggregate_psth_analysis import AggregateSession

matplotlib.use('TkAgg')

from plot_funcs import format_axis, add_x_scale_bar, plot_decoder_accuracy


def analyze_all_v_all_stats(
    dec_acc_df: pd.DataFrame,
    group_type: str = 'name',
    categories: List[str] = ['A', 'B', 'C', 'D'],
    chance_level: float = 0.25,
    parametric: bool = True,
) -> pd.DataFrame:
    """Calculates whether decoding accuracy is significantly better than chance

    (both empirical shuffle and theoretical chance) per category and overall.

    Parameters
    ----------
    dec_acc_df : pd.DataFrame
        DataFrame containing decoding accuracy columns (e.g. 'A_data',
        'A_shuffle', ...).
    group_type : str
        Level/column to aggregate by before testing ('name' for animals, 'sess'
        for sessions).
    categories : List[str]
        List of decoded category labels.
    chance_level : float
        Theoretical chance level (default 0.25 for 4 classes).
    parametric : bool
        If True, uses paired t-test (`ttest_rel`). If False, uses Wilcoxon
        signed-rank test (`wilcoxon`).

    Returns
    -------
    pd.DataFrame
        Summary table with mean accuracies, test statistics, raw p-values, and
        Bonferroni-adjusted p-values.
    """
    # 1. Aggregate data by specified level (e.g., animal name or session)
    if group_type in dec_acc_df.index.names:
        df_grouped = dec_acc_df.groupby(level=group_type).mean()
    elif group_type in dec_acc_df.columns:
        df_grouped = dec_acc_df.groupby(group_type).mean()
    else:
        df_grouped = dec_acc_df.copy()

    results = []
    n_groups = len(df_grouped)

    # 2. Test each individual stimulus category
    for cat in categories:
        data_col = f'{cat}_data'
        shuf_col = f'{cat}_shuffle'

        if data_col not in df_grouped.columns or shuf_col not in df_grouped.columns:
            continue

        acc_data = df_grouped[data_col].values
        acc_shuf = df_grouped[shuf_col].values

        # Paired test vs. empirical shuffle (one-tailed: data > shuffle)
        if parametric:
            res_shuf = ttest_rel(acc_data, acc_shuf, alternative='greater')
            stat_shuf, p_shuf = res_shuf.statistic, res_shuf.pvalue
            stat_name = 't_stat'
        else:
            res_shuf = wilcoxon(
                acc_data, acc_shuf, alternative='greater', zero_method='wilcox'
            )
            stat_shuf, p_shuf = res_shuf.statistic, res_shuf.pvalue
            stat_name = 'W_stat'

        # 1-sample test vs. theoretical chance (one-tailed: data > chance_level)
        res_theo = ttest_1samp(acc_data, chance_level, alternative='greater')

        results.append({
            'Category': cat,
            'Mean_Data': np.mean(acc_data),
            'Mean_Shuffle': np.mean(acc_shuf),
            'Data_std': np.std(acc_data, ddof=1),
            f'vs_Shuffle_{stat_name}': stat_shuf,
            'vs_Shuffle_p_raw': p_shuf,
            'vs_Chance_p_raw': res_theo.pvalue,
        })

    # 3. Test Overall Mean across all categories
    all_data_cols = [f'{cat}_data' for cat in categories if f'{cat}_data' in df_grouped.columns]
    all_shuf_cols = [f'{cat}_shuffle' for cat in categories if f'{cat}_shuffle' in df_grouped.columns]

    overall_data = df_grouped[all_data_cols].mean(axis=1).values
    overall_shuf = df_grouped[all_shuf_cols].mean(axis=1).values

    if parametric:
        overall_res_shuf = ttest_rel(overall_data, overall_shuf, alternative='greater')
        stat_overall, p_overall_shuf = overall_res_shuf.statistic, overall_res_shuf.pvalue
    else:
        overall_res_shuf = wilcoxon(overall_data, overall_shuf, alternative='greater')
        stat_overall, p_overall_shuf = overall_res_shuf.statistic, overall_res_shuf.pvalue

    overall_res_theo = ttest_1samp(overall_data, chance_level, alternative='greater')

    results.append({
        'Category': 'OVERALL',
        'Mean_Data': np.mean(overall_data),
        'Mean_Shuffle': np.mean(overall_shuf),
        'Data_std': np.std(overall_data, ddof=1),
        f'vs_Shuffle_{stat_name}': stat_overall,
        'vs_Shuffle_p_raw': p_overall_shuf,
        'vs_Chance_p_raw': overall_res_theo.pvalue,
    })

    stats_df = pd.DataFrame(results).set_index('Category')

    # 4. Apply Bonferroni correction for individual categories (excluding OVERALL)
    n_cats = len(categories)
    stats_df['vs_Shuffle_p_bonf'] = np.minimum(
        stats_df['vs_Shuffle_p_raw'] * n_cats, 1.0
    )
    # Don't multiply OVERALL p-value by n_cats
    stats_df.loc['OVERALL', 'vs_Shuffle_p_bonf'] = stats_df.loc[
        'OVERALL', 'vs_Shuffle_p_raw'
    ]

    stats_df['Signif_vs_Shuffle'] = stats_df['vs_Shuffle_p_bonf'] < 0.05

    return stats_df

def run_rm_anova_decoding(
    dec_acc_df: pd.DataFrame,
    group_type: str = 'name',
    categories: List[str] = ['A', 'B', 'C', 'D'],
    padjust: str = 'holm',
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs a 2-Way Repeated Measures ANOVA (Condition x Category)

    and post-hoc pairwise comparisons on decoding accuracy data.

    Parameters
    ----------
    dec_acc_df : pd.DataFrame
        Wide-format DataFrame containing columns like 'A_data', 'A_shuffle',
        etc.
    group_type : str
        Subject level/column to group by ('name' for animals or 'sess' for
        sessions).
    categories : List[str]
        List of decoded category names.
    padjust : str
        Method for multi-comparison p-value adjustment in post-hoc tests
        ('holm', 'fdr_bh', 'bonf').

    Returns
    -------
    aov_table : pd.DataFrame
        ANOVA summary table (F-statistics, p-values, partial eta-squared).
    post_hoc : pd.DataFrame
        Pairwise comparisons (Data vs. Shuffle per category with adjusted
        p-values).
    df_long : pd.DataFrame
        Reshaped long-format DataFrame used for the analysis.
    """
    # 1. Aggregate to the specified group level (e.g., animal or session)
    if group_type in dec_acc_df.index.names:
        df_grouped = dec_acc_df.groupby(level=group_type).mean()
    elif group_type in dec_acc_df.columns:
        df_grouped = dec_acc_df.groupby(group_type).mean()
    else:
        df_grouped = dec_acc_df.copy()

    # 2. Reshape from wide to long format
    df_long = df_grouped.reset_index().melt(
        id_vars=[group_type], var_name='cat_cond', value_name='accuracy'
    )

    # Split 'A_data' into category='A' and condition='data'
    df_long[['category', 'condition']] = df_long['cat_cond'].str.rsplit(
        '_', n=1, expand=True
    )

    # Filter for target categories
    df_long = df_long[df_long['category'].isin(categories)].copy()

    # 3. Two-Way Repeated Measures ANOVA
    # Factors: Within-subject factor 1 = condition (data vs shuffle), Factor 2 = category (A, B, C, D)
    aov_table = pg.rm_anova(
        data=df_long,
        dv='accuracy',
        within=['condition', 'category'],
        subject=group_type,
        detailed=True,
    )

    # 4. Post-hoc Pairwise Comparisons (Data vs. Shuffle per category)
    post_hoc = pg.pairwise_tests(
        data=df_long,
        dv='accuracy',
        within=['category', 'condition'],
        subject=group_type,
        alternative='greater',  # <-- Specify directional test (Data > Shuffle)
        padjust=padjust,  # 'holm'
        effsize='hedges',
    )

    # Filter post-hoc table to highlight Data vs. Shuffle comparisons per category
    post_hoc_filtered = post_hoc[
        post_hoc['Contrast'] == 'category * condition'
    ].copy()

    return aov_table, post_hoc_filtered, df_long

def plot_and_analyze_decoding(
        dec_dfs_path: Union[str, Path, List[Union[str, Path]], pd.DataFrame],
        plot_xlbls: Optional[List[str]] = None,
        combs: Optional[List[Tuple[int, int]]] = None,
        group_type: str = 'name',
        ylim: Optional[Tuple[float, float]] = None,
        ylabel: str = 'Decoding accuracy',
        figsize: Tuple[float, float] = (2.5, 2.5),
        save_path: Optional[Union[str, Path]] = None,
        format_axis_fn: Optional[callable] = format_axis,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Groups decoding accuracy data, generates a paired bar/scatter plot,
    performs paired t-tests on specified column pairs, and saves the plot.

    Parameters
    ----------
    dec_dfs_path : Union[str, Path, List[Union[str, Path]], pd.DataFrame]
        DataFrame or path(s) to HDF5 file(s) containing decoding accuracy.
    plot_xlbls : Optional[List[str]]
        Custom labels for the X-axis / bars. Defaults to DataFrame columns if None.
    combs : Optional[List[Tuple[int, int]]]
        List of index pairs (e.g., [[0, 1], [2, 3]]) to run t-tests between.
    group_type : str
        Level name or column to group by (e.g., 'name' or 'sess'). Default is 'name'.
    ylim : Optional[Tuple[float, float]]
        Limits for the Y-axis (bottom, top).
    ylabel : str
        Label for the Y-axis. Default is 'Decoding accuracy'.
    figsize : Tuple[float, float]
        Width and height of the figure in inches.
    save_path : Optional[Union[str, Path]]
        File path to save the output figure (e.g., PDF or PNG).
    format_axis_fn : Optional[callable]
        Axis formatting function applied to the `Axes` object.

    Returns
    -------
    fig, ax, grouped_df
        The matplotlib Figure, Axes object, and the grouped mean DataFrame.
    """
    # 1. Load data if path provided, or use DataFrame directly
    if isinstance(dec_dfs_path, pd.DataFrame):
        df = dec_dfs_path.copy()
    elif isinstance(dec_dfs_path, (str, Path)):
        df = pd.read_hdf(dec_dfs_path)
    elif isinstance(dec_dfs_path, list):
        dfs = [pd.read_hdf(p) if isinstance(p, (str, Path)) else p for p in dec_dfs_path]
        df = pd.concat(dfs, axis=1)
    else:
        raise TypeError("dec_dfs_path must be a DataFrame, Path, str, or list of Paths.")

    # 2. Group by specified index level or column
    if group_type in df.index.names:
        grouped_df = df.groupby(level=group_type).mean()
    elif group_type in df.columns:
        grouped_df = df.groupby(group_type).mean()
    else:
        grouped_df = df.copy()

    # 3. Determine labels
    if plot_xlbls is None:
        plot_xlbls = list(grouped_df.columns)

    grouped_vals = grouped_df.values  # Shape: (n_groups, n_conditions)
    mean_vals = grouped_df.mean().values  # Overall mean across groups

    # 4. Plot using strict Figure and Axes methods
    fig, ax = plt.subplots()
    fig.set_size_inches(*figsize)
    fig.set_layout_engine('tight')

    # Bar plot for overall means
    ax.bar(plot_xlbls, mean_vals, fc='w', ec='k')

    # Scatter individual group/subject data points
    for lbl, col_data in zip(plot_xlbls, grouped_vals.T):
        ax.scatter([lbl] * len(col_data), col_data, c='k', alpha=0.75)

    # Line plot connecting points across conditions per subject/group
    for subject_vals in grouped_vals:
        ax.plot(plot_xlbls, subject_vals, c='grey', lw=0.5)

    # Apply axis formatting and properties
    if format_axis_fn is not None:
        format_axis_fn(ax)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_ylabel(ylabel)

    # 5. Run independent t-tests for specified combinations
    if combs is not None:
        print(f"\n--- Paired T-Tests (Grouped by '{group_type}') ---")
        for comb in combs:
            idx1, idx2 = comb
            cond1 = plot_xlbls[idx1]
            cond2 = plot_xlbls[idx2]

            # stat, pval = ttest_ind(grouped_vals[:, idx1], grouped_vals[:, idx2], alternative='greater')
            stat, pval = ttest_rel(grouped_vals[:, idx1], grouped_vals[:, idx2],alternative='greater')
            print(f"[{cond1}] vs [{cond2}]: t = {stat:.4f}, p = {pval:.4e}")

    # 6. Save and display figure via methods
    if save_path is not None:
        fig.savefig(save_path)

    fig.show()

    return fig, ax, grouped_df


def sim_across_sessions(sess_list: list, _resps_dict: dict, pips2comp: list, x_ser: np.ndarray, window: list[float, float]):
    """
    Computes the cosine similarity of neural responses between specified stimulus
    pairs across a sequence of sessions.

    Parameters
    ----------
    sess_list : list of str
        A list of session identifiers (keys) to process in order.
    _resps_dict : dict
        A nested dictionary containing population responses, structured as
        _resps_dict[session][pip]. Response arrays can be 2D (units x time)
        or 3D (trials x units x time).
    pips2comp : list of str
        A list containing exactly two stimulus/pip identifiers (e.g., ['A-0', 'B-0'])
        whose population response vectors will be compared.
    x_ser : np.ndarray
        An array of time values corresponding to the time axis of the response matrices.
    window : list of float
        A two-element list specifying the [start, end] time window (in seconds)
        over which responses are averaged.

    Returns
    -------
    resp_sim_by_sess : dict
        A dictionary mapping each session identifier to an array of cosine similarity
        values comparing the reference stimulus to subsequent stimuli across the sessions.
    """
    window_idxs = [np.where(x_ser == t)[0][0] for t in window]
    resps_by_pip = {}
    for pip in pips2comp:
        resps = [_resps_dict[sess][pip] for sess in sess_list]
        resps = [np.mean(resp, axis=0) if resp.ndim == 3 else resp for resp in resps]
        resp_vectors = [resp[:, window_idxs[0]:window_idxs[1]].mean(axis=-1) for resp in resps]
        resps_by_pip[pip] = resp_vectors

    resp_sim_by_sess = {}
    for sess_i, sess in enumerate(sess_list):
        sim_mat = cosine_similarity([resps_by_pip[pip][sess_i] for pip in pips2comp])
        resp_sim_by_sess[sess] = sim_mat[0, 1:]

    return resp_sim_by_sess


# =========================================================================
# Analysis for sorted PSTH plot
# =========================================================================
plot_config_path = Path(r"H:\ephys\psth_plot_config_ephys_2401_2504.yaml")
aggr_savedir = Path(r"X:\Dammy\aggr_savedir")
figdir = Path(r"X:\Dammy\figures\decoding_plots_aggr_sessions_ephys_2401_2504")

# Set up stylesheet and line styling
plt.style.use('figure_stylesheet.mplstyle')

# Load plot configuration
plot_config = {}
assert plot_config_path.is_file()
with open(plot_config_path, 'r') as f:
    plot_config = yaml.safe_load(f)


# Dummy args structure expected by AggregateSession
class DummyArgs:
    event_responses_pkl = str(aggr_savedir / "dummy.pkl")


args = DummyArgs()

# Initialize AggregateSession for reproducing Figure 5b (sorted PSTH matrix)
# Using ['A-0', 'X'] or custom pips matching the original main() workflow
pips_to_plot = ['A-0', 'X']
dev_ABAD_dev_ABCD = AggregateSession(None, args, plot_config, pips_to_plot)

# Load or aggregate response data
dev_ABAD_dev_ABCD.aggregate_mean_sess_responses(
    concat_savename=aggr_savedir / 'trial_stim_responses.joblib',
    reload_save=True
)

# =========================================================================
# Figure 3b: Replicating Sorted PSTH Matrix and Mean Time Series Panel
# =========================================================================
plot_config['psth_plot_kwargs']['plot_window'] = [-0.25, 1.0]
dev_ABAD_dev_ABCD.plot_sorted_psth_mat(plot_kwargs=plot_config['psth_plot_kwargs'])
window = plot_config['window']

# Generate and format standalone plots for each pip/stimulus category
for pip in pips_to_plot:
    if f'{pip}_sorted_psth' not in dev_ABAD_dev_ABCD.plots:
        continue

    fig_3b, axes_3b,_ = dev_ABAD_dev_ABCD.plots[f'{pip}_sorted_psth']

    # Format axes, scale bars, and dimensions to match Fig. 5b layout precisely
    add_x_scale_bar(
        axes_3b[1], size=0.2, label='0.2 s', color='k',
        frameon=False, fontproperties={'size': 5}
    )
    axes_3b[0].locator_params(axis='y', nbins=2)
    fig_3b.set_size_inches(1.2, 1.8)
    fig_3b.set_layout_engine('tight')

    fig_3b.show()


# =========================================================================
# Figure 3c: Stim decoding
# =========================================================================

# Prepare dataframe
stim_dec_df: pd.DataFrame = pd.read_hdf(figdir / r"stim_decoding_df.h5")
stim_dec_df.set_index('name', append=True, inplace=True)

# Run function
fig_3c, ax_3c, stim_dec_grouped = plot_and_analyze_decoding(
    dec_dfs_path=stim_dec_df,
    plot_xlbls=[f'{e}_{ei}' for ei in range(int(len(stim_dec_df.columns)/2)) for e in ['data', 'shuffle'] ],
    combs=[[0, 1],[2,3],[0,2]],
    group_type='name',
    ylim=(0.4, 1),
    figsize=(3, 2.5),
    save_path= None
)
format_axis(ax_3c,hlines=[0.5])
fig_3c.show()
stim_dec_df_by_name = stim_dec_df.groupby('name').mean()
stim_dec_df_by_name_paired = stim_dec_df_by_name.values.reshape(len(stim_dec_df_by_name), 2, 2).transpose(1, 0, 2)
[print(ttest_rel(_arr[:,0],_arr[:,1])) for _arr in stim_dec_df_by_name_paired]

# =========================================================================
# Figure 3d: Pip vs Pip confusion matrix
# =========================================================================

for _cm_path, _dec_df_path,sffx in zip([r"all_v_all_decoding_cm.npy",r"all_v_all_train_rare_decoding_cm.npy"][:],
                                    [r"all_v_all_decoding_df.h5",r"all_v_all_train_rare_decoding_df.h5"],
                                    ["","_train_rare"]):


    print(f"\n--- {_dec_df_path} ---")
    cm_path = figdir / _cm_path
    cms = np.load(cm_path)
    cms_shuf = np.load(cm_path.with_stem(f'{cm_path.stem}_shuffle') )
    dec_df: pd.DataFrame = pd.read_hdf(figdir / _dec_df_path )
    sessnames = dec_df.index

    names = [sess.split('_')[0] for sess in sessnames]
    unique_names = set(names)
    dec_acc = {lbl: np.diagonal(_cm,axis1=1, axis2=2) for lbl,_cm in zip(['data','shuffle'],[cms,cms_shuf])}
    dec_acc_list = [dec_acc[lbl][:,i] for i,_ in enumerate('ABCD') for lbl in dec_acc]
    dec_lbl_list = [f'{lbl}' for i,pip in enumerate('ABCD') for lbl in dec_acc]
    dec_cols = [f'{pip}_{lbl}' for lbl in dec_acc for i,pip in enumerate('ABCD')]
    dec_acc_df = pd.concat([pd.DataFrame(accs) for accs in dec_acc.values()],axis=1)
    dec_acc_df.columns = dec_cols
    dec_acc_df['name'] = names
    dec_acc_df.index = dec_df.index
    dec_acc_df.set_index('name',append=True,inplace=True)
    dec_acc_df = dec_acc_df[[f'{pip}_{lbl}' for i,pip in enumerate('ABCD') for lbl in dec_acc]]

    for group_type in ['sess','name']:
        grouped_acc = dec_acc_df.groupby(group_type).mean()
        dec_acc_plot = plot_decoder_accuracy(grouped_acc.values.T,labels=dec_lbl_list,n_features=4)
        dec_acc_plot[1].get_legend().remove()
        dec_acc_plot[1].set_yticks(np.arange(0,1,0.25))
        dec_acc_plot[1].set_ylim(0,0.9)
        dec_acc_plot[1].set_ylabel('Decoding accuracy')
        dec_acc_plot[0].set_size_inches(4,2.5)
        dec_acc_plot[0].show()
        dec_acc_plot[0].savefig(figdir/f"all_v_all_acc_by_{group_type}{sffx}.pdf")
    # Execute ANOVA on your dec_acc_df
    aov_table, post_hoc, df_long = run_rm_anova_decoding(
        dec_acc_df, group_type='name', padjust='holm'
    )
    aov_table = aov_table.round(4)
    post_hoc = post_hoc.round(4)

    # 1. Extract values for Figure 3d (All vs All Decoding ANOVA & Post-hoc)
    print("\n=== EXTRACTED STATS FOR MANUSCRIPT  ===")
    for index, row in aov_table.iterrows():
        source = row['Source']
        f_val = row['F']
        p_val = row['p-unc'] if 'p-unc' in row else row.get('p-GG-corr', np.nan)
        eta_sq = row.get('ng2', np.nan)
        print(f"RM-ANOVA [{source}]: F = {f_val:.2f}, p = {p_val:.4f}, eta_g^2 = {eta_sq:.3f}")

    # Extract post-hoc stats for individual categories (A, B, C, D)
    for index, row in post_hoc.iterrows():
        cat = row['category']
        p_adj = row['p-corr']
        t_val = row['T']
        hedges_g = row['hedges']
        print(f"Post-hoc [Category {cat}]: t = {t_val:.2f}, p_adj = {p_adj:.4f}, Hedges' g = {hedges_g:.2f}")


# =========================================================================
# Figure 3e: Similarity over sessions analysis
# =========================================================================

# Load data and prepare structures
resps_dict = joblib.load(r"D:\ephys\rare_freq_by_pip_resps_ephys_2401_2504.joblib")
all_sess = list(resps_dict.keys())
all_names = np.unique([sess.split('_')[0] for sess in all_sess])

sess_by_name = {}
for name in all_names:
    animal_sessions = [sess for sess in all_sess if name in sess]
    sess_by_name[name] = sorted(animal_sessions)
all_names = ['DO79', 'DO80', 'DO81', 'DO95', 'DO96', 'DO97']

x_ser = np.linspace(-0.1, 0.25, 36)
plt.style.use('figure_stylesheet.mplstyle')

pips = ['A-0', 'B-0']
sess_num_cutoff = 8

# Create figure matching target layout for panel e
fig_3e, ax_3e = plt.subplots(figsize=(1.8, 2.5))

pip_comps = ['A-0', 'B-0']
sims_across_mice = []
for name in all_names[:]:
    sim_A2B_across_sessions = sim_across_sessions(sess_by_name[name], resps_dict, pip_comps, x_ser, [0.0, 0.25])
    sim_A2B = np.array(list(sim_A2B_across_sessions.values())[:sess_num_cutoff])[:sess_num_cutoff]
    sims_across_mice.append(sim_A2B[:])

sims_across_mice_arr = np.squeeze(np.array(sims_across_mice))

# Plot individual traces in light grey
for e in sims_across_mice_arr:
    ax_3e.plot(np.arange(1, len(e) + 1), e, c='k', alpha=0.2, lw=0.5)

# Plot mean trace in bold black
mean_trace = np.mean(sims_across_mice_arr, axis=0)
ax_3e.plot(np.arange(1, len(mean_trace) + 1), mean_trace, c='k', lw=2)

ax_3e.set_title('A to B', fontsize=8)
ax_3e.set_ylabel('Cosine similarity')
ax_3e.set_xlabel('Session')
ax_3e.set_xticks([1, 5])
ax_3e.set_ylim(0.3, 1.05)

# Add significance bar matching target image
ax_3e.annotate('', xy=(1, 1.02), xytext=(8, 1.02),
              xycoords='data', textcoords='data',
              arrowprops=dict(arrowstyle="-", lw=1, color='k'))
ax_3e.text(4.5, 1.04, '*', ha='center', va='bottom', fontsize=10)

format_axis(ax_3e)
fig_3e.set_layout_engine('tight')
fig_3e.show()

# Extract Kendall's Tau / Trend statistics for Figure 3e (Similarity Across Sessions)
session_indices = np.arange(sims_across_mice_arr.shape[1])
mean_sim_trace = np.mean(sims_across_mice_arr, axis=0)
tau_mean, p_val_mean = kendalltau(session_indices, mean_sim_trace)
print(f"Similarity across sessions (Mean Trace Trend) - Kendall's tau: {tau_mean:.4f}, p-val: {p_val_mean:.4e}")

# 3. Paired t-test between final and initial sessions for Figure 3e
t_rel_res = ttest_rel(sims_across_mice_arr[:, -1], sims_across_mice_arr[:, 0], alternative='greater')
print(f"Similarity across sessions (Final vs Initial Session t-test): t = {t_rel_res.statistic:.4f}, p = {t_rel_res.pvalue:.4e}")

# =========================================================================
# Figure 3f & g: Rare vs Frequent mean time series
# =========================================================================

# Initialize AggregateSession with identical event names
rare_freq_resps = AggregateSession(None, args, plot_config, ['A-0', 'A-0', 'A-0'])
rare_freq_resps.aggregate_mean_sess_responses(conds=['rare','frequent'],filt_by_prc=False, prc_thr=0.5,
                                              prc_pips=['A-0_rare','A-0_frequent'],prc_mutual=True,
                                              concat_savename=aggr_savedir/ 'A-0_rare_A-0_frequent.joblib',
                                              zscore_by_trial=False,
                                              reload_save=True)
rare_freq_resps.plot_mean_ts(figdir,)

from single_unit_analysis import plot_unit_resp
rare_freq_sessnames = rare_freq_resps.concatenated_sessnames
idxs_2_use = ['DO97' in sess for sess in rare_freq_sessnames ]

# Plot example units
x_ser = rare_freq_resps.x_ser
good_units = [1,3,9,23,26,28][:2]
rare_freq_examples_plot = plt.subplots(nrows=len(good_units),ncols=1,figsize=(1,1.5),sharex='all')
plot_unit_resp(rare_freq_examples_plot,rare_freq_resps.smoothed_concatenated_responses,rare_freq_sessnames,
               rare_freq_resps.smoothed_concatenated_responses_sem,
               x_ser,idx_mask=idxs_2_use,units=good_units, n_units=2)
[ax.axvline(0,ls='--',c='k') for ax in rare_freq_examples_plot[1].flatten()]
[ax.set_title('') for ax in rare_freq_examples_plot[1].flatten()]
[ax.set_xlim(-0.1,1.25) for ax in rare_freq_examples_plot[1].flatten()]
rare_freq_examples_plot[0].set_layout_engine('constrained')
rare_freq_examples_plot[0].show()

# =========================================================================
# Figure 3h: Rare vs Frequent Decoding
# =========================================================================

# Prepare combined dataframe
rf_dec_dfs = {
    n: pd.read_hdf(figdir / fn).set_index('name', append=True)
    for n, fn in zip(['reg'], ['rare_freq_decoding_using_post0_5_no_zscore_df.h5',])
}

rf_df_list = []
for n, df in rf_dec_dfs.items():
    df.columns = [f'{c}_{n}' for c in df.columns]
    rf_df_list.append(df)
rf_dec_df = pd.concat(rf_df_list, axis=1)

# Labels and t-test combinations
plot_x_lbls = [f'{e}_{ee.split("_")[-1]}' for e, ee in zip(['data', 'shuffle'] * len(rf_dec_dfs), rf_dec_df.columns)]
ttest_combs = [[0, 1],]

# Run function
fig_3h_1, ax_3h_1, rf_grouped = plot_and_analyze_decoding(
    dec_dfs_path=rf_dec_df,
    plot_xlbls=plot_x_lbls,
    combs=ttest_combs,
    group_type='name',
    ylim=(0.4, 0.65),
    figsize=(2.5, 2.5),
    save_path=None
)

fig_3h_1.show()

# =========================================================================
# Figure 3i & j: Normal vs Deviant mean time series
# =========================================================================
# Initialize AggregateSession with identical event names
norm_dev_resps = AggregateSession(None, args, plot_config, ['A-0', 'A-1'])
norm_dev_resps.aggregate_mean_sess_responses(
        filt_by_trial_num=True,
        pip_4_filt=['A-1'], pip_2_filt=['A-0'],
        filt_func=lambda tn, ft: [max(tn[tn < e] - e) + e for e in ft],
        # filt_func= lambda tn, ft: [e-1 for e in ft],
        concat_savename=aggr_savedir / 'A-0_A-1_1_trial_back.joblib',
        reload_save=True)

norm_dev_resps.plot_mean_ts(figdir,)

norm_dev_sessnames = norm_dev_resps.concatenated_sessnames
idxs_2_use = ['DO97' in sess for sess in norm_dev_sessnames ]

# Plot example units
x_ser = norm_dev_resps.x_ser
norm_dev_good_units = [0,2,20][1:]
normdev_examples_plot = plt.subplots(nrows=len(norm_dev_good_units),ncols=1,figsize=(1,1.5),sharex='all')
plot_unit_resp(normdev_examples_plot,norm_dev_resps.smoothed_concatenated_responses,norm_dev_sessnames,
               norm_dev_resps.smoothed_concatenated_responses_sem,
               x_ser,idx_mask=idxs_2_use,units=norm_dev_good_units, n_units=2)
[ax.axvline(0, ls='--', c='k') for ax in normdev_examples_plot[1].flatten()]
[ax.axvline(0.5, ls='--', c='r') for ax in normdev_examples_plot[1].flatten()]
[ax.set_title('') for ax in normdev_examples_plot[1].flatten()]
[ax.set_xlim(-0.1,1.25) for ax in normdev_examples_plot[1].flatten()]
normdev_examples_plot[0].set_layout_engine('constrained')
normdev_examples_plot[0].show()

# =========================================================================
# Figure 3k: Normal vs Deviant time resolved decoding
# =========================================================================
from aggregate_psth_analysis import padded_rolling_mean
from aggregate_ephys_funcs import decode_responses
from plot_funcs import format_axis

# 1. Setup paths and parameters

norm_dev_figdir = Path(r'X:\Dammy\figures\decoding_plots_aggr_sessions_ephys_2401_2504')
if not norm_dev_figdir.is_dir():
    norm_dev_figdir.mkdir(parents=True, exist_ok=True)

event_responses = joblib.load(r"D:\ephys\normdev_resps_ephys_2401_2504.joblib")

dec_over_time_window = [-0.5, 1.5]

window_size = 25
resp_width = list(event_responses.values())[0]['A-0'].shape[-1]
resp_x_ser = np.linspace(*dec_over_time_window, resp_width)

pips_as_ints = {pip: pip_i for pip_i, pip in enumerate([f'{pip}-{pip_i}' for pip in 'ABCD' for pip_i in range(2)])}
patt_is = [0, 1]
#
# 3. Sliding window decoding across sessions and time
normdev_ts_dec_dict = {}
bad_dec_sess = set()

for sessname, sess_data in tqdm(event_responses.items(), desc='Decoding across sessions'):
    for t in range(resp_width - window_size):
        xys = [
            (
                # sess_data[f'A-{pip_i}'][150:200] if pip_i == 0 else sess_data[f'A-{pip_i}'],
                sess_data[f'A-{pip_i}'][100:200][::3] if pip_i == 0 else sess_data[f'A-{pip_i}'],
                np.full_like(sess_data[f'A-{pip_i}'][:, 0, 0][100:200][::3] if pip_i ==0 else sess_data[f'A-{pip_i}'],
                             pips_as_ints[f'A-{pip_i}'])
            )
            for pip_i in patt_is
        ]

        ys = np.hstack([np.full(xy[0].shape[0], pips_as_ints[f'A-{pip_i}']) for xy, pip_i in zip(xys, patt_is)])
        xs = np.vstack([xy[0][:, :, t:t + window_size].mean(axis=-1) for xy in xys])

        try:
            normdev_ts_dec_dict[f'{sessname}-{t}:{t + window_size}s'] = decode_responses(
                xs, ys, dec_kwargs={'cv_folds': 5, 'n_runs': 10}
            )
        except ValueError:
            bad_dec_sess.add(sessname)
            continue

# 4. Process accuracy time-series into DataFrame
sess2use = [dec_name.split('-')[0] for dec_name in normdev_ts_dec_dict.keys()]
norm_dev_accs_ts_dict = {
    sessname: {
        t: np.mean(normdev_ts_dec_dict[f'{sessname}-{t}:{t + window_size}s']['data'].accuracy)
        for t in range(resp_width - window_size)
    }
    for sessname in tqdm(event_responses.keys(), desc='Aggregating accuracies') if sessname in sess2use
}

norm_dev_accs_ts_df = pd.DataFrame(norm_dev_accs_ts_dict).T
norm_dev_accs_ts_df.columns = np.round(resp_x_ser[window_size:], 2)
#
# from aggregate_ephys_funcs import decode_over_sliding_t
#
# norm_dev_accs_ts_df, norm_dev_accs_ts_dict = decode_over_sliding_t(event_responses, 0.25,
#                                                                    dec_over_time_window, pips_as_ints,
#                                                                    ['A-0', 'A-1'], )
norm_dev_accs_ts_df.columns = np.round(resp_x_ser[window_size:], 2)

# 5. Plotting results
fig_3k, ax_3k = plt.subplots(figsize=(2, 1.5))

# mean_trace = padded_rolling_mean(np.array(norm_dev_accs_ts_df), 1).mean(axis=0)
mean_trace = norm_dev_accs_ts_df.mean(axis=0)
x_cols = norm_dev_accs_ts_df.columns.tolist()

ax_3k.plot(x_cols, mean_trace, color='k')

lower = (norm_dev_accs_ts_df - norm_dev_accs_ts_df.sem(axis=0)).values
upper = (norm_dev_accs_ts_df + norm_dev_accs_ts_df.sem(axis=0)).values
ax_3k.fill_between(
    x_cols,
    lower.mean(axis=0),
    upper.mean(axis=0),
    color='k', alpha=0.1
)

ax_3k.set_xlabel('time (s)')
ax_3k.set_ylabel('decoding accuracy')
ax_3k.set_title('')

format_axis(ax_3k, hlines=[0.5], vspan=[[t, t + 0.15] for t in np.arange(0, 1, 0.25)])
ax_3k.axvspan(0.5, 0.5 + 0.15, fc='red', alpha=0.1)

fig_3k.set_layout_engine('tight')
fig_3k.show()

# =========================================================================
# Figure 3l: Normal vs Deviant decoding
# =========================================================================

# Prepare dataframe
normdev_dec_df: pd.DataFrame = pd.read_hdf(figdir / "normdev_decoding_0.5_1.5_df.h5")
normdev_dec_df.drop('DO79_240227a', axis=0, inplace=True)
normdev_dec_df.set_index('name', append=True, inplace=True)

# Run function
fig_3h_2, ax_3h_2, normdev_grouped = plot_and_analyze_decoding(
    dec_dfs_path=normdev_dec_df,
    plot_xlbls=['data', 'shuffle'],
    combs=[[0, 1]],
    group_type='name',
    ylim=(0.4, 0.65),
    figsize=(2, 2.5),
    save_path= None
)