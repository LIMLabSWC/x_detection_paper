from pathlib import Path
from copy import copy
from itertools import combinations
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from cycler import cycler

import statsmodels.formula.api as smf
from scipy.stats import ttest_ind, sem
from statsmodels.stats.multitest import multipletests

from aggregate_psth_analysis import AggregateSession
from plot_funcs import format_axis
# Hardcoded parameters
plot_config_path = Path(r"H:\ephys\abstraction_plot_config.yaml")
aggr_savedir = Path(r"X:\Dammy\aggr_savedir")
pips_for_dist = 'ABCDE'


# Set up stylesheet and line styling
plt.style.use('figure_stylesheet.mplstyle')
plt.rcParams['axes.prop_cycle'] = cycler(color=['k', '#e8739bff', '#73e8c1ff'])

# Load plot configuration
plot_config = {}
assert plot_config_path.is_file()
with open(plot_config_path, 'r') as f:
    plot_config = yaml.safe_load(f)

pkl_dir = aggr_savedir


# Dummy args structure expected by AggregateSession
class DummyArgs:
    event_responses_pkl = str(aggr_savedir / "dummy.pkl")


args = DummyArgs()

# Initialize AggregateSession with identical event names
dev_ABAD_dev_ABCD = AggregateSession(None, args, plot_config, ['A-0', 'A-1', 'A-2'])

# Replicate exact response aggregation, filtering, and trimming
dev_ABAD_dev_ABCD.aggregate_mean_sess_responses(
    conds=['normal', 'dev_ABCD1', 'dev_ABBA1'],
    concat_savename=aggr_savedir / 'A-0_A-1_A-2_trial_normed_trimmed_A-0.joblib',
    reload_save=True,
    zscore_flag=True,
    filt_by_trial_num=True,
    pip_2_filt=['A-0_normal'],
    pip_4_filt=['A-0_normal'],
    filt_func=lambda tn, ft: np.arange(100, 120)
)

# =========================================================================
# Figure 5b: dCA1 Population Responses (PSTH Mean Time Series)
# =========================================================================
fig_5b, ax_5b = plt.subplots(figsize=(1.2, 1.8))
dev_ABAD_dev_ABCD.plot_mean_ts(
    figdir=aggr_savedir,
    plot_smoothed_ts=False,
    figsize=(1.2, 1.8),
    ts_plot_window=[-0.1, 1.1]
)
fig_5b.show()

# =========================================================================
# PCA Processing & Pseudo-population Setup
# =========================================================================
pca_name = 'abstraction_pca_by_animal'
animal_per_sess = [sess.split('_')[0] for sess in dev_ABAD_dev_ABCD.concatenated_sessnames]
all_animals = np.unique([sess.split('_')[0] for sess in dev_ABAD_dev_ABCD.concatenated_sessnames])

for animal in all_animals:
    dev_ABAD_dev_ABCD.pca_pseudo_pop(pca_name=pca_name, by_animal=True, animal=animal)

# Reconstruct combined projected PCA dictionary by animal
proj_by_name = {}
for animal in all_animals:
    animal_proj = dev_ABAD_dev_ABCD.pca[f'{pca_name}_{animal}'].projected_pca_ts_by_cond['by_class']
    animal_proj = {f'{pip}_{animal}': animal_proj[pip] for pip in animal_proj}
    proj_by_name.update(animal_proj)

dev_ABAD_dev_ABCD.pca[pca_name] = dev_ABAD_dev_ABCD.pca[f'{pca_name}_{all_animals[0]}']
dev_ABAD_dev_ABCD.pca[pca_name].projected_pca_ts_by_cond = {'by_class': proj_by_name}

# =========================================================================
# Figure 5c: Cosine Distance from ABCD(0) across Pattern Positions
# =========================================================================
euc_dist_plot = dev_ABAD_dev_ABCD.plot_pca_euclidean_distance(
    pca_name=f'{pca_name}',
    times=[[t, t + 0.2] for t in np.arange(0.0, 1.25, 0.25)],
    figdir=aggr_savedir,
    n_pcs=15,
    align_trajs=True,
    align_pip_grouping=[-1],
    align_method="orthogonal",
    reference=None,
    ref_id=plot_config['abstraction_pca_by_animal']['plot_kwargs']['ref_id'],
    metric='cosine',
    reduce_when_pairwise="per_event_mean",
    labels_map=None,
    figsize=(4, 4),
    save_pdf=False,
    save_csv=False
)

# Extract events and matrices
dist_dict = euc_dist_plot[0]
n_conds = len(pips_for_dist)
dist_events_grouped = np.reshape(dist_dict['events'], (-1, n_conds)).tolist()

# Compute distances across pattern positions (A through E)
ref_event = 'A-0'
dists_by_times = []

for ti, _ in enumerate(list('ABCDE')):
    dists_by_animal = []
    for ai, animal_events in enumerate(dist_events_grouped):
        # Locate reference index for current animal
        ref_i = ai * n_conds + next(i for i, e in enumerate(animal_events) if ref_event in e)
        for ei, e in enumerate(animal_events):
            if ref_event not in e:
                dist_to_ref = dist_dict['distances'][ti][ref_i, ai * n_conds + ei]
                dists_by_animal.append(dist_to_ref)
    dists_by_times.append(dists_by_animal)

# Separate into paired conditions
dists_paired_by_event = [list(zip(*dists_by_times))[start::2] for start in [0, 1]]
dists_by_ref_event = [dists_paired_by_event]

# Plot distance trajectory across pattern positions
dist_over_t_plot = plt.subplots(figsize=(0.8, 1.8))
fig, ax = dist_over_t_plot

lbls = ['dev ABCD1', 'dev ABBA1']
cols = ['#e8739bff', '#73e8c1ff']

for ei, event_dists in enumerate(dists_paired_by_event):
    mean_vals = np.mean(event_dists, axis=0)
    sem_vals = sem(event_dists, axis=0, ddof=1)

    ax.errorbar(
        [f'{pip_i + 1}' for pip_i in range(5)],
        mean_vals,
        sem_vals,
        color=cols[ei],
        fmt='-o',
        capsize=5,
        lw=2,
        capthick=1,
        label=lbls[ei]
    )

ax.margins(x=0.1)
format_axis(ax, ylabel='Cosine distance', xlabel='Pattern position')
fig.show()

# =========================================================================
# Statistical Modeling & Analysis (LME & Corrections)
# =========================================================================
print("\n--- LINEAR MIXED-EFFECTS MODEL (LME) SUMMARY ---")
analysis_rows = []
for ei, event_dists in enumerate(dists_by_ref_event[0]):
    event_dists = np.array(event_dists)
    deviant_label = f'Condition_{ei + 1}'
    n_sessions, n_positions = event_dists.shape
    for sess_idx in range(n_sessions):
        animal_id = animal_per_sess[sess_idx]
        for pos_idx in range(n_positions):
            analysis_rows.append({
                'animal_id': animal_id,
                'session_id': f'session_{sess_idx}',
                'animal_session': f'{animal_id}_session_{sess_idx}', # Combined hierarchical group
                'pattern_position': str(pos_idx + 1),
                'deviant_type': deviant_label,
                'cosine_distance': event_dists[sess_idx, pos_idx]
            })

df = pd.DataFrame(analysis_rows)

# Mixed-effects interaction model grouped by both animal and session (nested/combined)
formula = "cosine_distance ~ C(deviant_type) * C(pattern_position)"
model = smf.mixedlm(formula, data=df, groups=df["animal_session"])
results = model.fit()

# Post-hoc comparisons per position with FDR & Bonferroni adjustments
positions = [str(i) for i in range(1, 6)]
raw_p_values = []

for pos in positions:
    cond1 = df.query('pattern_position==@pos & deviant_type=="Condition_1"')['cosine_distance'].values
    cond2 = df.query('pattern_position==@pos & deviant_type=="Condition_2"')['cosine_distance'].values
    t_stat, p_val = ttest_ind(cond1, cond2)
    raw_p_values.append(p_val)

reject_fdr, corrected_p_fdr, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')
reject_bonf, corrected_p_bonf, _, _ = multipletests(raw_p_values, alpha=0.05, method='bonferroni')

print("\n--- POST-HOC POSITION COMPARISONS ---")
for pos, raw_p, fdr_p, bonf_p in zip(positions, raw_p_values, corrected_p_fdr, corrected_p_bonf):
    print(f"Position {pos}: Raw p = {raw_p:.4f} | FDR adj p = {fdr_p:.4f} | Bonferroni adj p = {bonf_p:.4f} "
          f"{'***' if fdr_p < 0.05 else ''}")

# =========================================================================
# Automatic Extraction of LME & Post-hoc Statistics for Manuscript
# =========================================================================

print("\n--- EXTRACTED LME MODEL STATISTICS ---")
print(f"Model Converged: {results.converged}")

# Print coefficients, standard errors, z-values, and p-values for all terms
for param in results.params.index:
    coef = results.params[param]
    se = results.bse[param]
    z_val = results.tvalues[param]
    p_val = results.pvalues[param]
    print(f"Param [{param}]: beta = {coef:.3f}, se = {se:.3f}, z = {z_val:.2f}, p = {p_val:.4e}")

print("\n--- EXTRACTED POSITION-WISE COMPARISONS ---")
for pos, raw_p, bonf_p in zip(positions, raw_p_values, corrected_p_bonf):
    c1_vals = df.query('pattern_position==@pos & deviant_type=="Condition_1"')['cosine_distance'].values
    c2_vals = df.query('pattern_position==@pos & deviant_type=="Condition_2"')['cosine_distance'].values
    beta_diff = np.mean(c1_vals) - np.mean(c2_vals)

    print(f"Position {pos}: beta difference = {beta_diff:.3f} | Raw p = {raw_p:.4e} | Bonferroni adj p = {bonf_p:.4e}")
# =========================================================================
# Figure 5d: Neural Trajectories (1D PC Time Series for PC0 and PC1)
# =========================================================================
pc_1d_ts_plot = plt.subplots(nrows=2, sharex='all')
pc_kwargs = plot_config.get(pca_name, {})

for pc_i, pc in enumerate([0, 1]):
    pc_kwargs_copy = copy(pc_kwargs)
    pc_kwargs_copy.setdefault('plot_kwargs', {})['ax'] = pc_1d_ts_plot[1][pc_i]
    dev_ABAD_dev_ABCD.plot_1d_pca(pca_name, pc, aggr_savedir, pca_kwargs=pc_kwargs_copy)
    format_axis(pc_1d_ts_plot[1][pc_i], ylabel=f'PC {pc_i}')
    pc_1d_ts_plot[1][pc_i].set_xlim(-.1, 1.1)

pc_1d_ts_plot[0].subplots_adjust(hspace=0.1)
pc_1d_ts_plot[0].set_size_inches((1.6, 2))
pc_1d_ts_plot[0].show()

# =========================================================================
# Figure 5e: Population State Space (2D PC Scatter Windows: post pip 2 & post pip 4)
# =========================================================================
for pca_combs in [[0, 1]]:
    pc_scatter_by_t_plot = plt.subplots(ncols=2, sharex=False, sharey=True)
    for wi, win in enumerate([[0.4, 0.45], [0.9, 1]]):
        pc_kwargs_copy = copy(pc_kwargs)
        pc_kwargs_copy.setdefault('plot_kwargs', {})['ax'] = pc_scatter_by_t_plot[1][wi]
        pc_kwargs_copy['plot_kwargs']['markers'] = sum(
            [[m] * len(plot_config.get('pips_2_plot', pips_for_dist))
             for m in ['o', 's', 'P', '^', 'D'][:len(all_animals)]], []
        )
        dev_ABAD_dev_ABCD.scatter_pca(pca_name, win, pca_combs, aggr_savedir, pc_kwargs_copy)

    [ax.margins(x=0.1, y=0.1) for ax in pc_scatter_by_t_plot[1]]
    pc_scatter_by_t_plot[0].set_size_inches((1.6, 2))
    pc_scatter_by_t_plot[0].show()
