from pathlib import Path
from copy import copy

import h5py
import joblib
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.signal import lfilter
from tqdm import tqdm

from aggregate_psth_analysis import AggregateSession
# from xdetectioncore.plotting import format_axis
from plot_funcs import format_axis

# Hardcoded parameters (update as needed)
plot_config_path = Path(r"H:\ephys\abcd_vs_abba_config.yaml")
aggr_savedir = Path(r"X:\Dammy\aggr_savedir")

# Set up stylesheet and line styling
plt.style.use('figure_stylesheet.mplstyle')
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
    concat_savename=aggr_savedir / 'abcd_vs_abba_means.joblib',
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
fig_5g, axes_5g = plt.subplots(nrows=3, sharex='all', figsize=(1.6, 2.2))
pc_kwargs = plot_config.get(pca_name, {})

for pc_i, pc in enumerate([0, 1, 3]):
    pc_kwargs_copy = copy(pc_kwargs)
    pc_kwargs_copy.setdefault('plot_kwargs', {})['ax'] = axes_5g[pc_i]

    abcd_vs_abba.plot_1d_pca(pca_name, pc, figdir=aggr_savedir, pca_kwargs=pc_kwargs_copy)
    format_axis(axes_5g[pc_i], ylabel=f'PC {pc_i + 1}')
    axes_5g[pc_i].set_xlim(-0.1, 1.1)

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


# =========================================================================
# CCGP Core Functions & Null Shuffling Pipeline
# =========================================================================

def dir_metrics(Ptr, Ntr, Pte, Nte, balance_train=True, classifier="diaglda", sd_floor=1e-3, rng=None):
    """Calculates balanced accuracy and AUC for a given training/testing direction.
       Expected input shapes: (time, units, trials)
    """
    if rng is None:
        rng = np.random.default_rng()

    nT, nUnits, nTr_p = Ptr.shape
    _, _, nTr_n = Ntr.shape

    if balance_train:
        n = min(nTr_p, nTr_n)
        p_idx = rng.choice(nTr_p, n, replace=False)
        n_idx = rng.choice(nTr_n, n, replace=False)
        Ptr = Ptr[:, :, p_idx]
        Ntr = Ntr[:, :, n_idx]

    muP = np.mean(Ptr, axis=2)  # (nT, nUnits)
    muN = np.mean(Ntr, axis=2)  # (nT, nUnits)

    if classifier == "diaglda":
        combined = np.concatenate([Ptr, Ntr], axis=2)
        var_pooled = np.var(combined, axis=2)
        w = (muP - muN) / np.maximum(var_pooled, sd_floor ** 2)
    else:
        w = muP - muN

    b = -0.5 * np.sum((muP + muN) * w, axis=1, keepdims=True)  # (nT, 1)

    sP = np.sum(Pte * w[:, :, np.newaxis], axis=1) + b  # (nT, nTestP)
    sN = np.sum(Nte * w[:, :, np.newaxis], axis=1) + b  # (nT, nTestN)

    bal = 0.5 * (np.mean(sP >= 0, axis=1) + np.mean(sN < 0, axis=1))

    if sP.shape[1] < 2 or sN.shape[1] < 2:
        auc = np.full(nT, np.nan)
        return bal, auc

    Dm = sP[:, :, np.newaxis] - sN[:, np.newaxis, :]  # (nT, nTestP, nTestN)
    auc = np.mean(np.mean((Dm > 0) + 0.5 * (Dm == 0), axis=2), axis=1)

    return bal, auc


def ccgp_session_both(Fc, opt, rng, cons=['rule', 'pitch'], n_resamples=10):
    """Computes cross-condition generalization performance (CCGP) for rule and pitch."""
    # 0-indexed condition mapping: [0: A-0 (ABCD0), 1: A-1 (ABBA0), 2: A-2 (ABCD1), 3: A-3 (ABBA1)]
    dirs = {
        'rule': np.array([[0, 1, 2, 3], [2, 3, 0, 1]]),  # ABCD vs ABBA, across bands
        'pitch': np.array([[0, 2, 1, 3], [1, 3, 0, 2]])  # band0 vs band1, across sequence types
    }
    nT = Fc[0].shape[0]
    res = {}
    for con in cons:
        res[con] = {'balacc': np.zeros((nT, n_resamples)), 'auc': np.zeros((nT, n_resamples))}

    for r in range(n_resamples):
        tr = []
        te = []
        for c in range(4):
            n = Fc[c].shape[2]  # number of trials
            n_train = max(opt.get('minTrainPerCond', 5), int(opt.get('trainFrac', 0.7) * n))
            p = rng.permutation(n)
            tr.append(p[:n_train])
            te.append(p[n_train:])

        for con in cons:
            b = np.zeros(nT)
            a = np.zeros(nT)
            d_pairs = dirs[con]
            for d in range(2):
                q = d_pairs[d]
                bd, ad = dir_metrics(
                    Fc[q[0]][:, :, tr[q[0]]],
                    Fc[q[1]][:, :, tr[q[1]]],
                    Fc[q[2]][:, :, te[q[2]]],
                    Fc[q[3]][:, :, te[q[3]]],
                    balance_train=opt.get('balanceTrain', True),
                    classifier=opt.get('classifier', 'diaglda'),
                    sd_floor=opt.get('sdFloor', 1e-3),
                    rng=rng
                )
                b += bd / 2.0
                a += ad / 2.0
            res[con]['balacc'][:, r] = b
            res[con]['auc'][:, r] = a

    for con in cons:
        res[con]['balacc'] = np.mean(res[con]['balacc'], axis=1)
        res[con]['auc'] = np.mean(res[con]['auc'], axis=1)
    return res


def shuffle_pairs(Fc, pairs, rng):
    """Shuffle trials between the two conditions of each row of pairs (counts preserved)."""
    Fs = [f.copy() for f in Fc]
    for i in range(pairs.shape[0]):
        a, b = pairs[i, 0], pairs[i, 1]
        P = np.concatenate([Fc[a], Fc[b]], axis=2)
        n_total = P.shape[2]
        p = rng.permutation(n_total)
        na = Fc[a].shape[2]
        Fs[a] = P[:, :, p[:na]]
        Fs[b] = P[:, :, p[na:]]
    return Fs


def compute_ccgp_null(valid_session_ids, abcd_vs_abba_resp_dict, opt, baseline_mask):
    """Computes CCGP null distribution on label-shuffled data across valid sessions."""
    nS = len(valid_session_ids)
    nT = len(np.arange(opt['tStart'], opt['tEnd'] + opt['dt'], opt['dt']))
    nNull = opt.get('nNull', 50)
    nResNull = opt.get('nResamplesNull', opt['nResamples'])

    N = {
        con: {m: np.zeros((nS, nT, nNull), dtype=np.float32) for m in opt['metrics']}
        for con in opt['contrasts']
    }

    rule_pairs = np.array([[0, 1], [2, 3]])
    pitch_pairs = np.array([[0, 2], [1, 3]])

    print(f"\n--- Computing CCGP Null Distribution ({nS} sessions x {nNull} shuffles) ---")
    for s_idx, sessname in enumerate(tqdm(valid_session_ids, desc='Computing Null')):
        sess_data = abcd_vs_abba_resp_dict[sessname]

        Fc = []
        cond_keys = ['A-0', 'A-1', 'A-2', 'A-3']
        for c_key in cond_keys:
            X = sess_data[c_key].copy()
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

        sess_rng = np.random.default_rng(opt['seed'] + 1000 + s_idx)

        for k in range(nNull):
            Fc_rule_shuff = shuffle_pairs(Fc, rule_pairs, sess_rng)
            r1 = ccgp_session_both(Fc_rule_shuff, opt, sess_rng, cons=['rule'], n_resamples=nResNull)
            N['rule']['balacc'][s_idx, :, k] = r1['rule']['balacc']
            N['rule']['auc'][s_idx, :, k] = r1['rule']['auc']

            Fc_pitch_shuff = shuffle_pairs(Fc, pitch_pairs, sess_rng)
            r2 = ccgp_session_both(Fc_pitch_shuff, opt, sess_rng, cons=['pitch'], n_resamples=nResNull)
            N['pitch']['balacc'][s_idx, :, k] = r2['pitch']['balacc']
            N['pitch']['auc'][s_idx, :, k] = r2['pitch']['auc']

    return N


def draw_null_band(ax, t, nullCurves, opt):
    """Draws pointwise percentile null band behind data."""
    lo = np.percentile(nullCurves, opt['nullBand'][0], axis=1)
    hi = np.percentile(nullCurves, opt['nullBand'][1], axis=1)
    ok = np.isfinite(lo) & np.isfinite(hi)
    ax.fill_between(t[ok], lo[ok], hi[ok], color='gray', alpha=0.25, ec='none', zorder=0)


# =========================================================================
# CCGP Run Script: Pitch & Rule Decoding Over Time
# =========================================================================

# 1. Load pre-aggregated responses
all_resps_path = Path(r"D:\ephys\ABCD_ABBA_no_zscore_ephys_2401_2504.joblib")
abcd_vs_abba_resp_dict = joblib.load(all_resps_path)

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
    'nSessPerAnimal': 18,
    'sessSelect': 'last',
    'requireNSess': False,
    # --- Null Configuration ---
    'doNull': True,
    'nNull': 10,
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