from pathlib import Path
from copy import copy

import h5py
import joblib
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.stats import ttest_rel
from tqdm import tqdm

from analysis_scripts.aggregate_psth_analysis import AggregateSession
# from xdetectioncore.plotting import format_axis
from analysis_scripts.plot_funcs import format_axis
from analysis_scripts.rule_decoding_funcs import ccgp_session_both, compute_ccgp_null, draw_null_band, std_decode_session_both, \
    compute_std_decode_null, plot_time_resolved_decoding, plot_all_sessions_mean_sem, time_window_averaged_LDA, \
    rename_dict_keys, plot_axis_emergence, window_index_bounds, compute_axis_alignment_timecourse

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Hardcoded parameters (update as needed)
plot_config_path = DATA_DIR / "configs" / "fig05b_abcd_vs_abba.yaml"
aggr_savedir = DATA_DIR / "aggregate_cache"

# Set up stylesheet and line styling
plt.style.use('../figure_stylesheet.mplstyle')
plt.rcParams['axes.prop_cycle'] = cycler(color=['k', '#147d7dff', '#e8739bff', '#73e8c1ff'])

# Load plot configuration
plot_config = {}
if plot_config_path.is_file():
    with open(plot_config_path, 'r') as f:
        plot_config = yaml.safe_load(f)

pkl_dir = aggr_savedir


# Dummy args object expected by AggregateSession
class DummyArgs:
    event_responses_pkl = str(aggr_savedir / "dummy.pkl")


args = DummyArgs()
# Initialize AggregateSession
abcd_vs_abba = AggregateSession(None, args, plot_config, ['A-0', 'A-1', 'A-2', 'A-3'])
abcd_vs_abba.aggregate_mean_sess_responses(
    concat_savename=aggr_savedir / 'fig05b_abcd_vs_abba_response_aggregate.joblib',
    reload_save=True,
    zscore_flag=True,
)

pca_name = 'generalisation_pca_by_animal'
all_animals = np.unique([
    sess.split('_')[0] for sess in abcd_vs_abba.concatenated_sessnames
    if sess.split('_')[0] != 'DO82'
])

# Compute pseudo-population PCA by animal
for animal in all_animals:
    abcd_vs_abba.pca_pseudo_pop(pca_name=pca_name, by_animal=True, animal=animal, standardise=True)

# Combine projected trajectories across animals
proj_by_name = {}
for animal in all_animals:
    animal_proj = abcd_vs_abba.pca[f'{pca_name}_{animal}'].projected_pca_ts_by_cond['by_class']
    animal_proj = {f'{pip}_{animal}': animal_proj[pip] for pip in animal_proj}
    proj_by_name.update(animal_proj)

abcd_vs_abba.pca[pca_name] = abcd_vs_abba.pca[f'{pca_name}_{all_animals[0]}']
abcd_vs_abba.pca[pca_name].projected_pca_ts_by_cond = {'by_class': proj_by_name}

# =========================================================================
# Figure 5g: Neural Trajectories (1D PC Time Series for PC0, PC1, PC3)
# =========================================================================

fig_5g_plot_data = []

fig_5g, axes_5g = plt.subplots(nrows=3, sharex='all', figsize=(1.6, 2.2))
pc_kwargs = plot_config.get(pca_name, {})

for pc_i, pc in enumerate([0, 1, 3]):
    pc_kwargs_copy = copy(pc_kwargs)
    pc_kwargs_copy.setdefault('plot_kwargs', {})['ax'] = axes_5g[pc_i]

    abcd_vs_abba.plot_1d_pca(pca_name, pc, figdir=aggr_savedir, pca_kwargs=pc_kwargs_copy)
    format_axis(axes_5g[pc_i], ylabel=f'PC {pc_i + 1}')
    axes_5g[pc_i].set_xlim(-0.1, 1.1)

    projs = abcd_vs_abba.pca[pca_name].pca_ts_data
    proj_lbls_split = [[pc+1]+e.split('_')[-2:] for e in projs]
    multi_idx = pd.MultiIndex.from_tuples(proj_lbls_split,names=['pc','pip','animal'])
    projs_df = pd.DataFrame([np.squeeze(e) for e in projs.values()],index=multi_idx)
    mean_by_pip = projs_df.groupby('pip').mean()
    mean_by_pip.index = [e+f'_pc{pc+1}_mean' for e in mean_by_pip.index]

    fig_5g_plot_data.append(mean_by_pip)
    sem_by_pip = projs_df.groupby('pip').sem()
    sem_by_pip.index = [e+f'_pc{pc+1}_sem' for e in sem_by_pip.index]
    fig_5g_plot_data.append(sem_by_pip)
pd.concat(fig_5g_plot_data,axis=0).to_csv('fig_5g_plot_data.csv')

fig_5g.subplots_adjust(hspace=0.1)
fig_5g.show()

# =========================================================================
# Figure 5h: Population State Space (2D PC Scatter Windows: post pip 2 & post pip 4)
# =========================================================================
fig_5h, axes_5h = plt.subplots(ncols=2, sharex=False, sharey=True, figsize=(2.4, 2))
windows = [[0.4, 0.45], [1.0, 1.05]]

for wi, win in enumerate(windows):
    pc_kwargs_copy = copy(pc_kwargs)
    pc_kwargs_copy.setdefault('plot_kwargs', {})['ax'] = axes_5h[wi]
    pc_kwargs_copy['plot_kwargs']['markers'] = sum(
        [[m] * len(plot_config.get('pips_2_plot', [])) for m in ['o', 's', 'P', '^', 'D']], []
    )

    abcd_vs_abba.scatter_pca(pca_name, win, [0, 1], figdir=aggr_savedir, pca_kwargs=pc_kwargs_copy)

for ax in axes_5h:
    ax.margins(x=0.1)
    format_axis(ax)

axes_5h[0].set_title('post pip 2')
axes_5h[1].set_title('post pip 4')
fig_5h.set_layout_engine('constrained')
fig_5h.show()


pnts_list = []
for ax, lbl in zip(axes_5h.flatten(),['post_pip2','post_pip4']):
    for col in ax.collections:
        pts = col.get_offsets()  # shape (K, 2) -> [[x0, y0], [x1, y1], ...]
        pnts_list.append(
            {   "x": pts[0, 0],
                "y": pts[0, 1],
            })
lbls = [f'{win_lbl}_{animal}_{pip}' for win_lbl in ['post_pip_2','post_pip4']
        for pip in ['ABCD0', 'ABBA0','ABCD1','ABBA1'] for animal in all_animals]
pd.DataFrame(pnts_list,index=lbls).to_csv('fig_5h_plot_data.csv')


# Regular decoding:

# 1. Load pre-aggregated responses
all_resps_path = (
    DATA_DIR / "ephys_responses"
    / "fig05_abcd_vs_abba_responses_ephys_2401_2504.joblib"
)
abcd_vs_abba_resp_dict = joblib.load(all_resps_path)
out_dir = Path("figs")

plt.style.use(r'H:\ephys\figure_stylesheet.mplstyle')
plt.rcParams['axes.prop_cycle'] = cycler(color=['k', '#147d7dff', '#e8739bff', '#73e8c1ff'])

dt = 0.01
x_ser = np.round(np.arange(-0.5, 1.5 + dt, dt), 2)

pip_duration = 0.15
pip_separation = 0.1
pip_start_dict = {f'pip{pip_i + 1}': t for pip_i, t in enumerate(np.arange(0, 1, 0.25))}

data = abcd_vs_abba_resp_dict

seq_types = ['A-0', 'A-1', 'A-2', 'A-3']
key_to_pat_dict = {'A-0': 'ABCD0', 'A-1': 'ABBA0', 'A-2': 'ABCD1', 'A-3': 'ABBA1'}

mouse_sess_entries = list(data.keys())
data = rename_dict_keys(data, seq_types, key_to_pat_dict)
seq_types = list(key_to_pat_dict.values())

start_freqs = {'ABCD0': 0, 'ABBA0': 0, 'ABCD1': 1, 'ABBA1': 1}
pattern_types = {'ABCD0': 0, 'ABBA0': 1, 'ABCD1': 0, 'ABBA1': 1}
# assert False

axis_defs = {
    # 'pitch12': {'label': 'pitch', 'window': '12'},
    # 'pitch34': {'label': 'pitch', 'window': '34'},
    'pitch2': {'label': 'pitch', 'window': '4'},
    'pitch4': {'label': 'pitch', 'window': '4'},
    'pattern12': {'label': 'pattern', 'window': '12'},
    'pattern34': {'label': 'pattern', 'window': '34'},
    'pattern1': {'label': 'pattern', 'window': '1'},
    'pattern2': {'label': 'pattern', 'window': '2'},
    'pattern3': {'label': 'pattern', 'window': '3'},
    'pattern4': {'label': 'pattern', 'window': '4'},
}

windows = {
    '1': [np.where(x_ser == t)[0][0] for t in [0, 0.25]],
    '2': [np.where(x_ser == t)[0][0] for t in [0.25, 0.5]],
    '3': [np.where(x_ser == t)[0][0] for t in [0.5, 0.75]],
    '4': [np.where(x_ser == t)[0][0] for t in [0.75, 1]],
    '12': [np.where(x_ser == t)[0][0] for t in [0., 0.5]],
    '34': [np.where(x_ser == t)[0][0] for t in [0.5, 1.0]],
}

axis_names = ['pitch12', 'pitch34', 'pattern12', 'pattern34', 'pattern3', 'pattern4']

raw_seq_types = ["A-0", "A-1", "A-2", "A-3"]
key_to_pat_dict = {"A-0": "ABCD0", "A-1": "ABBA0", "A-2": "ABCD1", "A-3": "ABBA1"}

data = rename_dict_keys(data, raw_seq_types, key_to_pat_dict)

(    z_by_sess, angle_by_sess, acc_by_sess, cv_acc_by_sess,
     cv_shuff_acc_by_sess, z_trials_by_sess, z_tc_by_sess, z_tc_trials_by_sess,
) = time_window_averaged_LDA(
    data=data,
    seq_types=seq_types,
    sessions_by_id=mouse_sess_entries,
    start_freqs=start_freqs,
    pattern_types=pattern_types,
    axis_defs=axis_defs,
    windows=windows,
    k_folds=5,
    n_shuff=50,
    mode="mean",
    use_pca=False,
    shared_pca=True,
    pca_n_components=0.5,
    lda_shrinkage="auto",
    zscore_time=False,
    timecourse_t_range="full",
)

# 1. Rule axis emergence (ABBA vs. ABCD along pattern4 axis)
rule_axis_key = "pattern34"
rule_align_by_sess = compute_axis_alignment_timecourse(
    z_tc_trials_by_sess=z_tc_trials_by_sess,
    sessions_by_id=mouse_sess_entries,
    axis_name=rule_axis_key,
    pos_seqs=("ABBA0", "ABBA1"),
    neg_seqs=("ABCD0", "ABCD1"),
    align_positive=False,
    peak_window=window_index_bounds(x_ser, 0.75, 1.00),
)

fig_rule, ax_rule = plot_axis_emergence(
    x_ser=x_ser,
    alignment_by_sess=rule_align_by_sess,
    title="Rule axis emergence",
    ylabel="Axis alignment",
    pip_starts=tuple(pip_start_dict.values()),
    pip_duration=pip_duration,
    star_mode="none",
    save_path=out_dir / "rule_axis_emergence.pdf",
)
pd.DataFrame(rule_align_by_sess, index=mouse_sess_entries, columns=x_ser).to_csv(
    out_dir / "rule_axis_emergence.csv"
)

# 2. Pitch axis emergence (high pitch vs. low pitch along pitch2 axis)
pitch_axis_key = "pitch12"
pitch_align_by_sess = compute_axis_alignment_timecourse(
    z_tc_trials_by_sess=z_tc_trials_by_sess,
    sessions_by_id=mouse_sess_entries,
    axis_name=pitch_axis_key,
    pos_seqs=("ABCD1", "ABBA1"),
    neg_seqs=("ABCD0", "ABBA0"),
    align_positive=True,
    peak_window=window_index_bounds(x_ser, 0.25, 0.50),
)

fig_pitch, ax_pitch = plot_axis_emergence(
    x_ser=x_ser,
    alignment_by_sess=pitch_align_by_sess,
    title="Pitch axis emergence",
    ylabel="Axis alignment",
    pip_starts=tuple(pip_start_dict.values()),
    pip_duration=pip_duration,
    star_mode="none",
    save_path=out_dir / "pitch_axis_emergence.pdf",
)
pd.DataFrame(pitch_align_by_sess, index=mouse_sess_entries, columns=x_ser).to_csv(
    out_dir / "pitch_axis_emergence.csv"
)

panel_specs = [
        (0, 0, "pitch2", "Start pitch\n(pip 2)"),
        (0, 1, "pitch4", "Start pitch\n(pip 4)"),
        (1, 0, "pattern2", "ABCD vs ABBA\n(pip 2)"),
        (1, 1, "pattern4", "ABCD vs ABBA\n(pip 4)"),
    ]

fig, axs = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(2.4, 2.7),
    sharex="col",
    sharey="row",
    constrained_layout=True,
)
tests_res = []
for row_i, col_i, axis_key, title in panel_specs:
    ax = axs[row_i, col_i]

    data_vals = np.array([cv_acc_by_sess[axis_key][s] for s in mouse_sess_entries], dtype=float)
    shuff_vals = np.array([cv_shuff_acc_by_sess[axis_key][s] for s in mouse_sess_entries], dtype=float)
    mask = np.isfinite(data_vals) & np.isfinite(shuff_vals)
    data_vals = data_vals[mask]
    shuff_vals = shuff_vals[mask]

    bp = ax.boxplot(
        [data_vals, shuff_vals],
        positions=[0, 1],
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        # whis=(0, 100),
    )

    for box in bp["boxes"]:
        box.set_facecolor("white")
        box.set_edgecolor("k")
        box.set_linewidth(1)

    for median in bp["medians"]:
        median.set_color("#8b1e1e")
        median.set_linewidth(1.5)

    for whisker in bp["whiskers"]:
        whisker.set_color("k")
        whisker.set_linewidth(1)

    for cap in bp["caps"]:
        cap.set_color("k")
        cap.set_linewidth(1)

    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.7, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["data", "shuffle"])
    ax.set_title(title, fontsize=8)

    if col_i == 0:
        ax.set_ylabel("Decoding accuracy")

    if axis_key.startswith("pitch"):
        ax.set_ylim(0.4, 0.7)
        ax.set_yticks([0.4, 0.5, 0.6,])
    else:
        ax.set_ylim(0.4, 0.7)
        ax.set_yticks([0.4, 0.5, 0.6])

    # Paired test & significance bracket above data vs. shuffle
    _, p_val = ttest_rel(data_vals, shuff_vals, alternative="greater")
    tests_res.append(ttest_rel(data_vals, shuff_vals, alternative="greater"))
    sig_label = "*" if p_val < 0.05 else "ns"

    y_top = max(np.max(data_vals), np.max(shuff_vals))
    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    bracket_y = min(y_top + 0.04 * span, y_max - 0.08 * span)
    tick_h = 0.025 * span

    ax.plot(
        [0, 0, 1, 1],
        [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
        color="k",
        lw=0.8,
        clip_on=False,
    )
    ax.text(
        0.5,
        bracket_y + 0.01 * span,
        sig_label,
        ha="center",
        va="bottom",
        fontsize=8 if sig_label == "ns" else 10,
    )

    format_axis(ax)

fig.savefig(r"X:\Dammy\figures\LDA\lda_box_data_vs_shuffle.pdf", bbox_inches="tight")
fig.show()
plot_data = pd.concat([pd.DataFrame(cv_acc_by_sess),pd.DataFrame(cv_shuff_acc_by_sess)],axis=1)
plot_data.columns = [e if ci <4 else f'{e}_shuffle' for ci, e in enumerate(plot_data.columns)]
plot_data.to_csv('fig_5i_boxplots.csv')
### mega cv figure aggregated by mouse ###
animals = sorted(list({sess.split('_')[0] for sess in mouse_sess_entries}))

cv_acc_plot = plt.subplots(ncols=2, nrows=2, figsize=[2, 2.2], sharex='col', sharey='row')

for row_i, prop in enumerate(['pitch', 'pattern']):
    for col_i, win in enumerate(['2', '4']):
        ax = cv_acc_plot[1][row_i][col_i]
        axis_key = f'{prop}{win}'

        data_by_mouse = []
        shuff_by_mouse = []
        for animal in animals:
            anim_data = [cv_acc_by_sess[axis_key][sess] for sess in mouse_sess_entries if sess.startswith(animal)]
            anim_shuff = [cv_shuff_acc_by_sess[axis_key][sess] for sess in mouse_sess_entries if sess.startswith(animal)]
            data_by_mouse.append(np.mean(anim_data))
            shuff_by_mouse.append(np.mean(anim_shuff))

        data_by_mouse = np.array(data_by_mouse)
        shuff_by_mouse = np.array(shuff_by_mouse)

        ax.bar([0, 1], [data_by_mouse.mean(), shuff_by_mouse.mean()], width=0.6, fc='none', ec='k', lw=0.8)

        x_pts = [0, 1]
        for d_pt, s_pt in zip(data_by_mouse, shuff_by_mouse):
            ax.plot(x_pts, [d_pt, s_pt], color='gray', lw=0.5, zorder=1)
            ax.scatter(0, d_pt, color='k', s=12, alpha=0.7, zorder=2)
            ax.scatter(1, s_pt, color='k', s=12, alpha=0.7, zorder=2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['data', 'shuffle'])

        if 'pattern' in axis_key:
            ax.set_ylim(0.45, 0.55)
        else:
            ax.set_ylim(0.45, 0.6)
        format_axis(ax, hlines=[0.5])

        t_stat, p_val = ttest_rel(data_by_mouse, shuff_by_mouse, alternative='greater')
        print(f"[{axis_key}] Animal Paired t-test: t={t_stat:.3f}, p={p_val:.4e}")

[axs[0].set_ylabel('Decoding accuracy') for axs in cv_acc_plot[1]]
cv_acc_plot[0].set_layout_engine('tight')
cv_acc_plot[0].show()
cv_acc_plot[0].savefig(r'X:\Dammy\figures\LDA\lda_accuracy_vs_shuff_by_mouse.pdf')

### Per-animal rule decoding across windows 1, 2, 3, 4 (0.00-0.25, ..., 0.75-1.00) ---
rule_windows = ['pattern1', 'pattern2', 'pattern3', 'pattern4']
window_titles = [
    '0.00-0.25 s\npip1',
    '0.25-0.50 s\npip2',
    '0.50-0.75 s\npip3',
    '0.75-1.00 s\npip4'
]

fig, axs = plt.subplots(1, len(rule_windows), figsize=(2.2 * len(rule_windows), 2.5), sharey=True)
if len(rule_windows) == 1:
    axs = [axs]

for ax, axis_key, w_title in zip(axs, rule_windows, window_titles):
    x_ticks = []
    x_labels = []

    for idx, animal in enumerate(animals):
        anim_data = [cv_acc_by_sess[axis_key][sess] for sess in mouse_sess_entries if sess.startswith(animal)]
        anim_shuff = [cv_shuff_acc_by_sess[axis_key][sess] for sess in mouse_sess_entries if sess.startswith(animal)]

        if len(anim_data) == 0:
            continue

        mean_data = np.mean(anim_data)
        min_val, max_val = np.percentile(anim_shuff, 0), np.percentile(anim_shuff, 100)

        ax.bar(idx, max_val - min_val, bottom=min_val, width=0.4, color='lightgray', edgecolor='none')
        ax.scatter(idx, mean_data, color='k', s=16, zorder=3)

        x_ticks.append(idx)
        x_labels.append(animal)

    format_axis(ax, hlines=[0.5])
    ax.set_xticks(np.arange(len(animals)))
    ax.set_xticklabels([])
    ax.set_title(w_title, fontsize=8)
    ax.set_ylabel("Decoding accuracy (AUC)" if ax == axs[0] else "")

fig.suptitle("Per animal Rule decoding across sequence windows", y=1.02, fontsize=9)
fig.set_layout_engine('tight')
fig.show()
fig.savefig(r'X:\Dammy\figures\LDA\per_animal_rule_decoding_windows_1_4.pdf', bbox_inches='tight')

for axis_key in [f'pattern{i}' for i in '1234']:
    ### check rule 34 significance and individual animal t-tests against shuffle ###
    # axis_key = 'pattern4'
    data_by_mouse = []
    shuff_by_mouse = []

    print(f"--- Individual Animal Results ({axis_key}) ---")
    for animal in animals:
        anim_data = [cv_acc_by_sess[axis_key][sess] for sess in mouse_sess_entries if sess.startswith(animal)]
        anim_shuff = [cv_shuff_acc_by_sess[axis_key][sess] for sess in mouse_sess_entries if sess.startswith(animal)]

        mean_data = np.mean(anim_data)
        mean_shuff = np.mean(anim_shuff)

        data_by_mouse.append(mean_data)
        shuff_by_mouse.append(mean_shuff)

        if len(anim_data) > 1:
            t_anim, p_anim = ttest_rel(anim_data, anim_shuff, alternative='greater')
            print(f"Animal {animal}: data mean={mean_data:.4f}, shuff mean={mean_shuff:.4f} (paired t={t_anim:.3f}, p={p_anim:.4f})")
        else:
            print(f"Animal {animal}: data={mean_data:.4f}, shuff={mean_shuff:.4f} (single session)")

    data_by_mouse = np.array(data_by_mouse)
    shuff_by_mouse = np.array(shuff_by_mouse)

    t_stat, p_val = ttest_rel(data_by_mouse, shuff_by_mouse, alternative='greater')
    print(f"\n[rule_{axis_key}_by_animal] Group Animal Paired t-test: t={t_stat:.3f}, p={p_val:.4e}")



# =========================================================================
# CCGP Run Script: Pitch & Rule Decoding Over Time
# =========================================================================

# 2. Setup options and parameters
opt = {
    'dt': 0.01,
    'tStart': -0.5,
    'tEnd': 1.5,
    'integrateWin': 0.15,
    'integrateMode': 'causal',
    'smoothWin': 0.0,
    'doBaseline': True,
    'baselineWindow': [-0.50, 0.00],
    'classifier': 'diaglda',
    'balanceTrain': True,
    'trainFrac': 0.7,
    'nResamples': 10,
    'minTrainPerCond': 5,
    'minTestPerCond': 3,
    'sdFloor': 1e-3,
    'minTimeFrac': 0.90,
    'seed': 1,
    'nSessPerAnimal': 100,
    'sessSelect': 'last',
    'requireNSess': False,
    # --- Null Configuration ---
    'doNull': True,
    'nNull': 50,
    'nResamplesNull': 10,
    'nullBand': [2.5, 97.5],
    # --------------------------
    'contrasts': ['pitch', 'rule'],
    'metrics': ['balacc', 'auc'],
    'metricLabel': {'balacc': 'Balanced accuracy', 'auc': 'AUC'},
    'colors': {'pitch': [0.47, 0.27, 0.62], 'rule': [0.27, 0.56, 0.30]},
    'pipStarts': [0, 0.25, 0.50, 0.75],
    'pipDur': 0.15,
    # --- File paths for saving/loading ---
    'dataFile': aggr_savedir / "ccgp_timecourse_pitch_rule.h5",
    'nullFile': aggr_savedir / "ccgp_timecourse_pitch_rule_null.h5",
    'recompute': True,
    'recomputeNull': True,
}

t_ser = np.arange(-0.5, 1.5 + 0.01, 0.01)
nT = len(t_ser)

all_session_ids = list(abcd_vs_abba_resp_dict.keys())
rng = np.random.default_rng(opt['seed'])
baseline_mask = (t_ser >= opt['baselineWindow'][0]) & (t_ser < opt['baselineWindow'][1])

# 3. Session selection per animal
anim_to_all_sessions = {}
for sessname in all_session_ids:
    animal = sessname.split('_')[0]
    anim_to_all_sessions.setdefault(animal, []).append(sessname)

for animal in anim_to_all_sessions:
    anim_to_all_sessions[animal].sort()

selected_session_ids = []
session_counts_per_animal = {}

for animal, sess_list in anim_to_all_sessions.items():
    n_total = len(sess_list)
    if opt['sessSelect'] == 'random':
        perm = rng.permutation(n_total)
        ordered_sess = [sess_list[i] for i in perm]
    elif opt['sessSelect'] == 'first':
        ordered_sess = sess_list
    elif opt['sessSelect'] == 'last':
        ordered_sess = list(reversed(sess_list))
    else:
        raise ValueError("opt['sessSelect'] must be 'last', 'first', or 'random'.")

    cap = opt['nSessPerAnimal']
    chosen_sess = ordered_sess[:int(cap)] if np.isfinite(cap) else ordered_sess

    if opt['requireNSess'] and np.isfinite(cap) and len(chosen_sess) < cap:
        continue

    session_counts_per_animal[animal] = len(chosen_sess)
    selected_session_ids.extend(chosen_sess)

# 4. Main session-level processing loop
ccgp_results = {con: {m: np.nan * np.zeros((len(selected_session_ids), nT)) for m in opt['metrics']} for con in
                opt['contrasts']}
valid_sessions = []
animals = []

print("\n--- Running CCGP Time Course Analysis ---")
for s_idx, sessname in enumerate(tqdm(selected_session_ids, desc='Running CCGP across sessions')):
    sess_data = abcd_vs_abba_resp_dict[sessname]
    animal_name = sessname.split('_')[0]

    cond_keys = ['A-0', 'A-1', 'A-2', 'A-3']
    if not all(c in sess_data for c in cond_keys):
        continue

    Fc = []
    failed = False
    for c_key in cond_keys:
        X = sess_data[c_key]
        if X.ndim != 3:
            failed = True
            break

        if opt['doBaseline']:
            baseline_mean = np.mean(X[:, :, baseline_mask], axis=2, keepdims=True)
            X = X - baseline_mean

        integrate_win_s = opt.get('integrateWin', 0.0)
        if integrate_win_s > 0:
            dt = opt.get('dt', 0.01)
            int_bins = max(1, int(round(integrate_win_s / dt)))
            nT_bins = X.shape[2]
            X_int = np.zeros_like(X)
            for ti in range(nT_bins):
                start_idx = max(0, ti - int_bins + 1)
                X_int[:, :, ti] = np.mean(X[:, :, start_idx:ti + 1], axis=2)
            X = X_int

        X_transposed = np.transpose(X, (2, 1, 0))
        Fc.append(X_transposed)

    if failed or any(F.shape[2] < (opt['minTrainPerCond'] + opt['minTestPerCond']) for F in Fc):
        continue

    res = ccgp_session_both(Fc, opt, rng, cons=opt['contrasts'], n_resamples=opt['nResamples'])

    for con in opt['contrasts']:
        ccgp_results[con]['balacc'][s_idx, :] = res[con]['balacc']
        ccgp_results[con]['auc'][s_idx, :] = res[con]['auc']

    valid_sessions.append(sessname)
    animals.append(animal_name)

animals = np.array(animals)
for con in opt['contrasts']:
    for m in opt['metrics']:
        ccgp_results[con][m] = ccgp_results[con][m][:len(valid_sessions)]

print(f"Successfully processed {len(valid_sessions)} valid sessions across {len(np.unique(animals))} mice.")

fig_5m_plot_data = ccgp_results['rule']['auc']
pd.DataFrame(fig_5m_plot_data).to_csv('fig_5m_plot_data.csv')

# 5. Compute or Load Null Distribution via HDF5 if enabled
N = None
if opt['doNull']:
    if not opt['recomputeNull'] and Path(opt['nullFile']).is_file():
        with h5py.File(opt['nullFile'], 'r') as f:
            N = {
                con: {m: f[f"{con}/{m}"][()] for m in opt['metrics']}
                for con in opt['contrasts']
            }
        print(f"Loaded Null distribution from {opt['nullFile']}")
    else:
        N = compute_ccgp_null(valid_sessions, abcd_vs_abba_resp_dict, opt, baseline_mask)
        with h5py.File(opt['nullFile'], 'w') as f:
            for con in opt['contrasts']:
                for m in opt['metrics']:
                    f.create_dataset(f"{con}/{m}", data=N[con][m])
        print(f"Saved Null distribution to {opt['nullFile']}")

# 6. Plot Per-Animal Multi-Panel Grid with Null Bands
unique_animals = np.unique(animals)
nMice = len(unique_animals)
animal_to_sessions = {animal: np.where(animals == animal)[0] for animal in unique_animals}

for metric in opt['metrics']:
    fig, axes = plt.subplots(nrows=len(opt['contrasts']), ncols=nMice,
                             figsize=(2.2 * nMice, 3.2), sharex=True, constrained_layout=True)
    fig.suptitle(f"CCGP ({opt['metricLabel'][metric]}) by Animal | {len(valid_sessions)} sessions, {nMice} mice")

    row_axes = {con: [] for con in opt['contrasts']}

    for c_idx, con in enumerate(opt['contrasts']):
        data_matrix = ccgp_results[con][metric]

        for m_idx, animal in enumerate(unique_animals):
            ax = axes[c_idx, m_idx] if len(opt['contrasts']) > 1 else axes[m_idx]
            row_axes[con].append(ax)

            sess_indices = animal_to_sessions[animal]
            animal_data = data_matrix[sess_indices, :]

            # Draw Null Band if available (averaged across this animal's sessions)
            if N is not None:
                null_tensor = N[con][metric][sess_indices, :, :]  # (n_sess, nT, nNull)
                animal_null_mean = np.nanmean(null_tensor, axis=0)  # (nT, nNull)
                draw_null_band(ax, t_ser, animal_null_mean, opt)

            mean_trace = np.nanmean(animal_data, axis=0)
            n_valid_sess = np.sum(~np.isnan(animal_data).all(axis=1))
            sem_trace = np.nanstd(animal_data, axis=0) / np.sqrt(np.maximum(np.sum(~np.isnan(animal_data), axis=0), 1))

            ax.plot(t_ser, mean_trace, color=opt['colors'][con], lw=1.5)
            ax.fill_between(t_ser, mean_trace - sem_trace, mean_trace + sem_trace,
                            color=opt['colors'][con], alpha=0.25, ec='none')

            ax.axhline(0.5, color='k', linestyle=':', lw=0.8)

            if c_idx == 0:
                ax.set_title(f"{animal} (n={n_valid_sess})", fontsize=9)

            if m_idx == 0:
                ax.set_ylabel(f"{con.capitalize()}\n{opt['metricLabel'][metric]}")

            if c_idx == len(opt['contrasts']) - 1:
                ax.set_xlabel("Time (s)")

            ax.set_xlim(t_ser[0], t_ser[-1])

            vspans = [[p, p + opt['pipDur']] for p in opt['pipStarts']]
            format_axis(ax, vspan=vspans)

        row_ymin = min(ax.get_ylim()[0] for ax in row_axes[con])
        row_ymax = max(ax.get_ylim()[1] for ax in row_axes[con])
        for ax in row_axes[con]:
            ax.set_ylim(row_ymin, row_ymax)

    fig.show()

cv_folds = 5
dec_results = {
    con: {m: np.nan * np.zeros((len(selected_session_ids), nT)) for m in opt['metrics']}
    for con in opt['contrasts']
}
valid_sessions = []
animals = []

print("\n--- Running Standard Decoding Time Course Analysis ---")
for s_idx, sessname in enumerate(tqdm(selected_session_ids, desc='Running decoding across sessions')):
    sess_data = abcd_vs_abba_resp_dict[sessname]
    animal_name = sessname.split('_')[0]

    cond_keys = ['A-0', 'A-1', 'A-2', 'A-3']
    if not all(c in sess_data for c in cond_keys):
        continue

    Fc = []
    failed = False
    for c_key in cond_keys:
        X = sess_data[c_key]
        if X.ndim != 3:
            failed = True
            break

        if opt['doBaseline']:
            baseline_mean = np.mean(X[:, :, baseline_mask], axis=2, keepdims=True)
            X = X - baseline_mean

        integrate_win_s = opt.get('integrateWin', 0.0)
        if integrate_win_s > 0:
            dt = opt.get('dt', 0.01)
            int_bins = max(1, int(round(integrate_win_s / dt)))
            nT_bins = X.shape[2]
            X_int = np.zeros_like(X)
            for ti in range(nT_bins):
                start_idx = max(0, ti - int_bins + 1)
                X_int[:, :, ti] = np.mean(X[:, :, start_idx:ti + 1], axis=2)
            X = X_int

        Fc.append(np.transpose(X, (2, 1, 0)))

    if failed or any(F.shape[2] < cv_folds for F in Fc):
        continue

    res = std_decode_session_both(
        Fc,
        opt,
        rng,
        cons=opt['contrasts'],
        n_resamples=opt['nResamples'],
        cv_folds=cv_folds,
    )

    for con in opt['contrasts']:
        dec_results[con]['balacc'][s_idx, :] = res[con]['balacc']
        dec_results[con]['auc'][s_idx, :] = res[con]['auc']

    valid_sessions.append(sessname)
    animals.append(animal_name)

animals = np.array(animals)
for con in opt['contrasts']:
    for m in opt['metrics']:
        dec_results[con][m] = dec_results[con][m][:len(valid_sessions)]

N = None
if opt['doNull']:
    N = compute_std_decode_null(valid_sessions, abcd_vs_abba_resp_dict, opt, baseline_mask, cv_folds=cv_folds)

# Call after computing dec_results (and optional null N)
plot_time_resolved_decoding(
    t_ser=t_ser,
    dec_results=dec_results,
    animals=animals,
    opt=opt,
    null_results=N,
    save_dir=Path(''),
)

plot_all_sessions_mean_sem(
    t_ser=t_ser,
    dec_results=dec_results,
    animals=animals,
    opt=opt,
    null_results=N,
    save_dir=Path(''),
)

pd.DataFrame(dec_results['rule']['auc']).to_csv(f'fig_5l_rule_decodability.csv')
pd.DataFrame(N['rule']['auc'].mean(axis=0)).to_csv(f'fig_5l_rule_decodability_null.csv')
