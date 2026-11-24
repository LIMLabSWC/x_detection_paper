from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp, zscore
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from LDA.LDA import data
from plot_funcs import format_axis


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


def build_contrast_xy(Fc, contrast, balance_classes=True, rng=None):
    """
    Build feature matrix X (n_trials, nT, nUnits) and binary labels y (n_trials,)
    combining both pairs of conditions for `contrast` ('rule' or 'pitch').
    Fc is a list of 4 arrays ['A-0', 'A-1', 'A-2', 'A-3'], each with shape (nT, nUnits, nTrials).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Transpose each condition from (nT, nUnits, nTrials) -> (nTrials, nT, nUnits)
    cond_trials = [np.transpose(F, (2, 0, 1)) for F in Fc]

    if contrast == 'rule':
        # Class 0: ABCD (A-0, A-2); Class 1: ABBA (A-1, A-3)
        x0 = np.concatenate([cond_trials[0], cond_trials[2]], axis=0)
        x1 = np.concatenate([cond_trials[1], cond_trials[3]], axis=0)
    elif contrast == 'pitch':
        # Class 0: band0 (A-0, A-1); Class 1: band1 (A-2, A-3)
        x0 = np.concatenate([cond_trials[0], cond_trials[1]], axis=0)
        x1 = np.concatenate([cond_trials[2], cond_trials[3]], axis=0)
    else:
        raise ValueError(f"Unknown contrast '{contrast}'. Use 'rule' or 'pitch'.")

    if balance_classes:
        n_bal = min(x0.shape[0], x1.shape[0])
        if n_bal == 0:
            raise ValueError(f"Empty class in contrast '{contrast}'.")
        idx0 = rng.choice(x0.shape[0], size=n_bal, replace=False)
        idx1 = rng.choice(x1.shape[0], size=n_bal, replace=False)
        x0 = x0[idx0]
        x1 = x1[idx1]

    X = np.concatenate([x0, x1], axis=0)  # (n_trials, nT, nUnits)
    y = np.concatenate([
        np.zeros(x0.shape[0], dtype=int),
        np.ones(x1.shape[0], dtype=int),
    ])
    return X, y


def score_diaglda_timecourse(X_time, y_tr, X_te_time, y_te, classifier='diaglda', sd_floor=1e-3):
    """
    Fit classifier at each time bin on X_time (n_tr, nT, nUnits) and evaluate
    balanced accuracy and AUC on X_te_time (n_te, nT, nUnits).
    """
    nT = X_time.shape[1]
    bal = np.zeros(nT, dtype=float)
    auc = np.full(nT, np.nan, dtype=float)

    pos_tr = X_time[y_tr == 1]  # (nP_tr, nT, nUnits)
    neg_tr = X_time[y_tr == 0]  # (nN_tr, nT, nUnits)

    muP = np.mean(pos_tr, axis=0)  # (nT, nUnits)
    muN = np.mean(neg_tr, axis=0)  # (nT, nUnits)

    if classifier == 'diaglda':
        comb_tr = np.concatenate([pos_tr, neg_tr], axis=0)
        var_pooled = np.var(comb_tr, axis=0)  # (nT, nUnits)
        w = (muP - muN) / np.maximum(var_pooled, sd_floor ** 2)
    else:
        w = muP - muN

    b = -0.5 * np.sum((muP + muN) * w, axis=1, keepdims=True)  # (nT, 1)

    # Scores: (n_samples, nT)
    scores_te = np.einsum('itu,tu->it', X_te_time, w) + b.T
    preds_te = scores_te >= 0.0

    pos_mask = y_te == 1
    neg_mask = y_te == 0
    if np.any(pos_mask) and np.any(neg_mask):
        tpr = np.mean(preds_te[pos_mask], axis=0)
        tnr = np.mean(~preds_te[neg_mask], axis=0)
        bal = 0.5 * (tpr + tnr)

        sP = scores_te[pos_mask]  # (nP_te, nT)
        sN = scores_te[neg_mask]  # (nN_te, nT)
        # Pairwise comparison across test positives and negatives per time bin
        diff = sP[:, None, :] - sN[None, :, :]  # (nP_te, nN_te, nT)
        auc = np.mean((diff > 0) + 0.5 * (diff == 0), axis=(0, 1))

    return bal, auc


def std_decode_session_both(Fc, opt, rng, cons=['rule', 'pitch'], n_resamples=10, cv_folds=5):
    """
    Standard decoding trained on both categories/conditions and tested on both
    categories/conditions across stratified CV folds and resamples.
    """
    nT = Fc[0].shape[0]
    res = {
        con: {
            'balacc': np.zeros((nT, n_resamples), dtype=float),
            'auc': np.zeros((nT, n_resamples), dtype=float),
        }
        for con in cons
    }

    for con in cons:
        X, y = build_contrast_xy(
            Fc,
            contrast=con,
            balance_classes=opt.get('balanceTrain', True),
            rng=rng,
        )

        class_counts = np.bincount(y)
        n_splits = min(cv_folds, int(class_counts.min()))
        if n_splits < 2:
            for m in ('balacc', 'auc'):
                res[con][m][:] = np.nan
            continue

        for r in range(n_resamples):
            fold_seed = int(rng.integers(0, 2**31 - 1))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)

            fold_bal = np.zeros((nT, n_splits), dtype=float)
            fold_auc = np.zeros((nT, n_splits), dtype=float)

            for f_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
                bd, ad = score_diaglda_timecourse(
                    X[tr_idx],
                    y[tr_idx],
                    X[te_idx],
                    y[te_idx],
                    classifier=opt.get('classifier', 'diaglda'),
                    sd_floor=opt.get('sdFloor', 1e-3),
                )
                fold_bal[:, f_idx] = bd
                fold_auc[:, f_idx] = ad

            res[con]['balacc'][:, r] = np.nanmean(fold_bal, axis=1)
            res[con]['auc'][:, r] = np.nanmean(fold_auc, axis=1)

        res[con]['balacc'] = np.nanmean(res[con]['balacc'], axis=1)
        res[con]['auc'] = np.nanmean(res[con]['auc'], axis=1)

    return res


def compute_std_decode_null(valid_session_ids, abcd_vs_abba_resp_dict, opt, baseline_mask, cv_folds=5):
    """Computes label-shuffled null distribution for standard two-category decoding."""
    nS = len(valid_session_ids)
    nT = len(np.arange(opt['tStart'], opt['tEnd'] + opt['dt'], opt['dt']))
    nNull = opt.get('nNull', 50)
    nResNull = opt.get('nResamplesNull', opt['nResamples'])

    N = {
        con: {m: np.zeros((nS, nT, nNull), dtype=np.float32) for m in opt['metrics']}
        for con in opt['contrasts']
    }

    cond_keys = ['A-0', 'A-1', 'A-2', 'A-3']

    for s_idx, sessname in enumerate(tqdm(valid_session_ids, desc='Computing Standard Null')):
        sess_data = abcd_vs_abba_resp_dict[sessname]
        Fc = []
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

            Fc.append(np.transpose(X, (2, 1, 0)))

        sess_rng = np.random.default_rng(opt['seed'] + 1000 + s_idx)

        for con in opt['contrasts']:
            X_con, y_con = build_contrast_xy(Fc, contrast=con, balance_classes=False, rng=sess_rng)
            class_counts = np.bincount(y_con)
            n_splits = min(cv_folds, int(class_counts.min()))
            if n_splits < 2:
                N[con]['balacc'][s_idx, :, :] = np.nan
                N[con]['auc'][s_idx, :, :] = np.nan
                continue

            for k in range(nNull):
                y_shuff = sess_rng.permutation(y_con)
                fold_bal = np.zeros((nT, n_splits), dtype=float)
                fold_auc = np.zeros((nT, n_splits), dtype=float)

                fold_seed = int(sess_rng.integers(0, 2**31 - 1))
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)
                for f_idx, (tr_idx, te_idx) in enumerate(skf.split(X_con, y_shuff)):
                    bd, ad = score_diaglda_timecourse(
                        X_con[tr_idx],
                        y_shuff[tr_idx],
                        X_con[te_idx],
                        y_shuff[te_idx],
                        classifier=opt.get('classifier', 'diaglda'),
                        sd_floor=opt.get('sdFloor', 1e-3),
                    )
                    fold_bal[:, f_idx] = bd
                    fold_auc[:, f_idx] = ad

                N[con]['balacc'][s_idx, :, k] = np.nanmean(fold_bal, axis=1)
                N[con]['auc'][s_idx, :, k] = np.nanmean(fold_auc, axis=1)

    return N


def plot_time_resolved_decoding(
    t_ser,
    dec_results,
    animals,
    opt,
    null_results=None,
    save_dir=None,
):
    """
    Plot time-resolved decoding accuracy/AUC across all valid sessions
    and in a per-animal grid, matching the styling/conventions in final_fig_5_b.py.
    """
    valid_animals = np.unique(animals)
    n_mice = len(valid_animals)
    animal_to_idx = {animal: np.where(animals == animal)[0] for animal in valid_animals}
    vspans = [[p, p + opt["pipDur"]] for p in opt["pipStarts"]]

    for metric in opt["metrics"]:
        # 1. Summary time course across all sessions per contrast
        fig_all, axs_all = plt.subplots(
            nrows=len(opt["contrasts"]),
            ncols=1,
            figsize=(3.2, 1.8 * len(opt["contrasts"])),
            sharex=True,
            constrained_layout=True,
        )
        if len(opt["contrasts"]) == 1:
            axs_all = [axs_all]

        for c_idx, con in enumerate(opt["contrasts"]):
            ax = axs_all[c_idx]
            mat = dec_results[con][metric]  # (n_valid_sessions, nT)

            if null_results is not None:
                # Average null across sessions first -> (nT, nNull), then compute percentile band
                null_sess_mean = np.nanmean(null_results[con][metric], axis=0)
                lo = np.percentile(null_sess_mean, opt["nullBand"][0], axis=1)
                hi = np.percentile(null_sess_mean, opt["nullBand"][1], axis=1)
                ok = np.isfinite(lo) & np.isfinite(hi)
                ax.fill_between(
                    t_ser[ok],
                    lo[ok],
                    hi[ok],
                    color="gray",
                    alpha=0.25,
                    ec="none",
                    zorder=0,
                    label="Null band",
                )

            mean_trace = np.nanmean(mat, axis=0)
            n_valid = np.sum(~np.isnan(mat), axis=0)
            sem_trace = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(n_valid, 1))

            ax.plot(t_ser, mean_trace, color=opt["colors"][con], lw=1.5, label=con.capitalize())
            ax.fill_between(
                t_ser,
                mean_trace - sem_trace,
                mean_trace + sem_trace,
                color=opt["colors"][con],
                alpha=0.25,
                ec="none",
            )
            ax.axhline(0.5, color="k", linestyle=":", lw=0.8)
            ax.set_ylabel(f"{con.capitalize()}\n{opt['metricLabel'][metric]}")
            ax.set_xlim(t_ser[0], t_ser[-1])
            format_axis(ax, vspan=vspans)

        axs_all[-1].set_xlabel("Time (s)")
        fig_all.suptitle(
            f"Time-Resolved Decoding ({opt['metricLabel'][metric]}) | {len(animals)} sessions, {n_mice} mice",
            fontsize=9,
        )

        # 2. Per-animal grid (rows = contrasts, cols = animals)
        fig_grid, axes_grid = plt.subplots(
            nrows=len(opt["contrasts"]),
            ncols=n_mice,
            figsize=(2.2 * n_mice, 3.2),
            sharex=True,
            constrained_layout=True,
        )
        fig_grid.suptitle(
            f"Time-Resolved Decoding ({opt['metricLabel'][metric]}) by Animal | {len(animals)} sessions, {n_mice} mice",
            fontsize=9,
        )

        row_axes = {con: [] for con in opt["contrasts"]}
        for c_idx, con in enumerate(opt["contrasts"]):
            mat = dec_results[con][metric]
            for m_idx, animal in enumerate(valid_animals):
                ax = axes_grid[c_idx, m_idx] if len(opt["contrasts"]) > 1 else axes_grid[m_idx]
                row_axes[con].append(ax)

                s_idx = animal_to_idx[animal]
                anim_mat = mat[s_idx, :]

                if null_results is not None:
                    anim_null_mean = np.nanmean(null_results[con][metric][s_idx, :, :], axis=0)
                    lo = np.percentile(anim_null_mean, opt["nullBand"][0], axis=1)
                    hi = np.percentile(anim_null_mean, opt["nullBand"][1], axis=1)
                    ok = np.isfinite(lo) & np.isfinite(hi)
                    ax.fill_between(
                        t_ser[ok],
                        lo[ok],
                        hi[ok],
                        color="gray",
                        alpha=0.25,
                        ec="none",
                        zorder=0,
                    )

                mean_trace = np.nanmean(anim_mat, axis=0)
                n_valid_sess = int(np.sum(~np.isnan(anim_mat).all(axis=1)))
                n_valid_t = np.sum(~np.isnan(anim_mat), axis=0)
                sem_trace = np.nanstd(anim_mat, axis=0) / np.sqrt(np.maximum(n_valid_t, 1))

                ax.plot(t_ser, mean_trace, color=opt["colors"][con], lw=1.5)
                ax.fill_between(
                    t_ser,
                    mean_trace - sem_trace,
                    mean_trace + sem_trace,
                    color=opt["colors"][con],
                    alpha=0.25,
                    ec="none",
                )
                ax.axhline(0.5, color="k", linestyle=":", lw=0.8)

                if c_idx == 0:
                    ax.set_title(f"{animal} (n={n_valid_sess})", fontsize=9)
                if m_idx == 0:
                    ax.set_ylabel(f"{con.capitalize()}\n{opt['metricLabel'][metric]}")
                if c_idx == len(opt["contrasts"]) - 1:
                    ax.set_xlabel("Time (s)")

                ax.set_xlim(t_ser[0], t_ser[-1])
                format_axis(ax, vspan=vspans)

            row_ymin = min(ax.get_ylim()[0] for ax in row_axes[con])
            row_ymax = max(ax.get_ylim()[1] for ax in row_axes[con])
            for ax in row_axes[con]:
                ax.set_ylim(row_ymin, row_ymax)

        if save_dir is not None:
            out_path = Path(save_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            fig_all.savefig(out_path / f"time_resolved_decoding_summary_{metric}.pdf", bbox_inches="tight")
            fig_grid.savefig(out_path / f"time_resolved_decoding_by_animal_{metric}.pdf", bbox_inches="tight")

        fig_all.show()
        fig_grid.show()

def plot_all_sessions_mean_sem(
    t_ser,
    dec_results,
    animals,
    opt,
    null_results=None,
    save_dir=None,
):
    """
    Create a separate figure per metric showing mean +/- SEM across all valid sessions
    over time for each contrast.
    """
    valid_animals = np.unique(animals)
    n_mice = len(valid_animals)
    vspans = [[p, p + opt["pipDur"]] for p in opt["pipStarts"]]

    for metric in opt["metrics"]:
        fig_all, axs_all = plt.subplots(
            nrows=len(opt["contrasts"]),
            ncols=1,
            figsize=(3.5, 2.0 * len(opt["contrasts"])),
            sharex=True,
            constrained_layout=True,
        )
        if len(opt["contrasts"]) == 1:
            axs_all = [axs_all]

        for c_idx, con in enumerate(opt["contrasts"]):
            ax = axs_all[c_idx]
            mat = dec_results[con][metric]  # (n_valid_sessions, nT)

            if null_results is not None:
                null_sess_mean = np.nanmean(null_results[con][metric], axis=0)  # (nT, nNull)
                lo = np.percentile(null_sess_mean, opt["nullBand"][0], axis=1)
                hi = np.percentile(null_sess_mean, opt["nullBand"][1], axis=1)
                ok = np.isfinite(lo) & np.isfinite(hi)
                ax.fill_between(
                    t_ser[ok],
                    lo[ok],
                    hi[ok],
                    color="gray",
                    alpha=0.25,
                    ec="none",
                    zorder=0,
                    label="Null band",
                )

            mean_trace = np.nanmean(mat, axis=0)
            n_valid = np.sum(~np.isnan(mat), axis=0)
            sem_trace = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(n_valid, 1))

            ax.plot(t_ser, mean_trace, color=opt["colors"][con], lw=1.5, label=con.capitalize())
            ax.fill_between(
                t_ser,
                mean_trace - sem_trace,
                mean_trace + sem_trace,
                color=opt["colors"][con],
                alpha=0.25,
                ec="none",
            )

            ax.axhline(0.5, color="k", linestyle=":", lw=0.8)
            ax.set_ylabel(f"{con.capitalize()}\n{opt['metricLabel'][metric]}")
            ax.set_xlim(t_ser[0], t_ser[-1])
            format_axis(ax, vspan=vspans)

        axs_all[-1].set_xlabel("Time (s)")
        fig_all.suptitle(
            f"All Sessions Mean ± SEM ({opt['metricLabel'][metric]}) | {len(animals)} sessions, {n_mice} mice",
            fontsize=9,
        )

        if save_dir is not None:
            out_path = Path(save_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            fig_all.savefig(
                out_path / f"all_sessions_mean_sem_{metric}.pdf",
                bbox_inches="tight",
            )

        fig_all.show()

    def compute_axis_alignment_timecourse(
            z_tc_trials_by_sess,
            sessions_by_id,
            axis_name,
            pos_seqs,
            neg_seqs,
            align_positive=True,
            peak_window=None,
    ):
        """
        Compute per-session alignment timecourse: shape (n_sessions, n_time).
        At each timepoint, alignment is mean(z[pos_seqs]) - mean(z[neg_seqs]).
        If `align_positive=True`, flips a session's sign when its mean alignment
        over `peak_window` (or full timecourse) is negative.
        """
        alignments = []

        for sess in sessions_by_id:
            sess_tc = z_tc_trials_by_sess[axis_name][sess]

            pos_trials = np.concatenate([np.asarray(sess_tc[s]) for s in pos_seqs], axis=0)
            neg_trials = np.concatenate([np.asarray(sess_tc[s]) for s in neg_seqs], axis=0)

            align_tc = pos_trials.mean(axis=0) - neg_trials.mean(axis=0)

            if align_positive:
                ref_slice = slice(None) if peak_window is None else slice(peak_window[0], peak_window[1])
                if np.nanmean(align_tc[ref_slice]) < 0:
                    align_tc = -align_tc

            alignments.append(align_tc)

        return np.asarray(alignments, dtype=float)


    def plot_axis_emergence(
            x_ser,
            alignment_by_sess,
            title="Axis emergence",
            ylabel="Axis alignment",
            pip_starts=(0.0, 0.25, 0.5, 0.75),
            pip_duration=0.15,
            star_mode="none",
            alpha_p=0.05,
            xlim=(-0.35, 1.45),
            ylim=None,
            yticks=None,
            figsize=(2.2, 2.2),
            save_path=None,
    ):
        """
        Plot mean +/- SEM axis alignment timecourse using Figure and Axes methods.
        `star_mode`: 'none', 'peak' (red star at peak of mean curve), or
                     'sig' (red stars where one-sample t-test vs. 0 < alpha_p).
        """
        mean_tc = np.nanmean(alignment_by_sess, axis=0)
        n_sess = alignment_by_sess.shape[0]
        sem_tc = np.nanstd(alignment_by_sess, axis=0, ddof=1) / np.sqrt(n_sess)

        fig, ax = plt.subplots(figsize=figsize)

        for t0 in pip_starts:
            ax.axvspan(t0, t0 + pip_duration, color="#efefef", lw=0.0, zorder=0)

        ax.axvline(0.0, color="k", linestyle="--", linewidth=1.1, zorder=1)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1.5, zorder=2)

        ax.fill_between(
            x_ser,
            mean_tc - sem_tc,
            mean_tc + sem_tc,
            color="lightgray",
            alpha=0.7,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(x_ser, mean_tc, color="k", linewidth=2.0, zorder=3)

        if star_mode in ("peak", "sig"):
            y_top = np.nanmax(mean_tc + sem_tc)
            star_y = y_top + 0.08 * max(abs(y_top), 0.5)

            if star_mode == "peak":
                peak_idx = int(np.nanargmax(mean_tc))
                ax.scatter(
                    [x_ser[peak_idx]],
                    [star_y],
                    marker="*",
                    s=85,
                    facecolor="#d61f1f",
                    edgecolor="k",
                    linewidth=0.6,
                    zorder=5,
                )
            elif star_mode == "sig":
                _, p_vals = ttest_1samp(alignment_by_sess, popmean=0.0, axis=0, nan_policy="omit")
                sig_mask = p_vals < alpha_p
                if np.any(sig_mask):
                    ax.scatter(
                        x_ser[sig_mask],
                        np.full(np.sum(sig_mask), star_y),
                        marker="*",
                        s=70,
                        facecolor="#d61f1f",
                        edgecolor="k",
                        linewidth=0.6,
                        zorder=5,
                    )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        if yticks is not None:
            ax.set_yticks(yticks)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["bottom"].set_linewidth(1.1)
        ax.tick_params(axis="both", direction="in", length=5, width=1.1, top=False, right=False)

        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=600)

        return fig, ax


    def window_index_bounds(x_ser, t_start, t_end):
        idx = np.where((x_ser >= t_start) & (x_ser <= t_end))[0]
        if len(idx) == 0:
            return None
        return int(idx[0]), int(idx[-1]) + 1


def replace_key_recursive(d, old_key, new_key):
    if isinstance(d, dict):
        keys = list(d.keys())
        for key in keys:
            if key == old_key:
                d[new_key] = d.pop(old_key)
            replace_key_recursive(d.get(key, {}), old_key, new_key)
    elif isinstance(d, list):
        for item in d:
            replace_key_recursive(item, old_key, new_key)


def rename_dict_keys(d, seq_types, key_to_pat_dict):
    for seq_type in seq_types:
        old_key = seq_type
        new_key = key_to_pat_dict[seq_type]
        replace_key_recursive(data, old_key, new_key)
    return d


def plot_session_LDA_scatter_1D(
        z_trials_by_sess,
        session_key,
        axis_name,
        axis_defs,
        start_freqs,
        pattern_types,
        seq_types,
):
    if axis_defs[axis_name]['label'] == 'pitch':
        class_map = start_freqs
        xlabel = 'Pitch'
    else:
        class_map = pattern_types
        xlabel = 'Pattern'

    fig, ax = plt.subplots(figsize=(3, 3))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for i, seq in enumerate(seq_types):
        z_trials = z_trials_by_sess[axis_name][session_key][seq]
        y_val = class_map[seq]

        ax.scatter(
            np.full_like(z_trials, y_val),
            z_trials,
            alpha=0.7,
            color=colors[i],
            label=f"{seq}"
        )

    ax.axhline(0, color='k', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("LDA projection")
    ax.set_title(f"{axis_name} — {session_key}")
    ax.legend(frameon=False)
    fig.tight_layout(rect=[0.05, 0, 1, 0.96])
    fig.savefig(f"figs/scatter_{session_key}.svg", bbox_inches="tight", dpi=600)
    fig.savefig(f"figs/scatter_{session_key}.pdf", bbox_inches="tight", dpi=600)
    plt.close(fig)


def project_timecourse_onto_axis(data_single_sess, w, b, zscore_time=True):
    X = data_single_sess.copy()

    if zscore_time:
        X = zscore(X, axis=2)
        X[np.isnan(X)] = 0.0

    z_trials_time = np.tensordot(X, w, axes=([1], [0])) + b
    z_mean_time = z_trials_time.mean(axis=0)

    return z_trials_time, z_mean_time


def time_window_averaged_LDA(
        data,
        seq_types,
        sessions_by_id,
        start_freqs,
        pattern_types,
        axis_defs,
        windows,
        k_folds=5,
        n_shuff=100,
        mode="mean",
        step=1,
        lags=(0, -1, -2),
        lag_agg="mean",
        zscore_time=True,
        baseline_window=(0, 50),
        baseline_eps=1e-6,
        use_pca=False,
        shared_pca=True,
        pca_n_components=0.95,
        pca_whiten=False,
        lda_solver="lsqr",
        lda_shrinkage="auto",
        timecourse_t_range="full",
):
    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.decomposition import PCA
    from tqdm import tqdm

    def baseline_zscore(X, base0, base1, eps=1e-6):
        if base0 is None or base1 is None:
            raise ValueError("baseline_window must be a tuple (base0, base1).")
        base0 = int(base0)
        base1 = int(base1)

        n_time = X.shape[2]
        if not (0 <= base0 < base1 <= n_time):
            raise ValueError(f"Invalid baseline_window={baseline_window} for n_time={n_time}")

        mu = X[:, :, base0:base1].mean(axis=2, keepdims=True)
        sd = X[:, :, base0:base1].std(axis=2, keepdims=True)

        sd = np.where(sd < eps, 1.0, sd)
        Xn = (X - mu) / sd
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
        return Xn

    def make_temporal_X(data_single_sess, t0, t1):
        X = np.asarray(data_single_sess)
        if X.ndim != 3:
            raise ValueError(f"Expected (trials, neurons, time), got {X.shape}")

        n_trials, n_neurons, n_time = X.shape

        if not (0 <= t0 < t1 <= n_time):
            raise ValueError(f"Invalid window [{t0},{t1}) for n_time={n_time}")

        if zscore_time:
            b0, b1 = baseline_window
            X = baseline_zscore(X, b0, b1, eps=baseline_eps)

        if mode == "mean":
            return X[:, :, t0:t1].mean(axis=2)

        if mode == "stack":
            ts = np.arange(t0, t1, step, dtype=int)
            return X[:, :, ts].reshape(n_trials, n_neurons * len(ts))

        if mode == "lagged":
            ts = np.arange(t0, t1, step, dtype=int)
            lag_arr = np.array(lags, dtype=int)

            idx = ts[None, :] + lag_arr[:, None]
            valid_mask = (idx >= 0) & (idx < n_time)
            keep_cols = np.all(valid_mask, axis=0)
            idx = idx[:, keep_cols]

            if idx.shape[1] == 0:
                raise ValueError(
                    f"No valid timepoints for window [{t0},{t1}) with lags={lags}."
                )

            feats_per_t = []
            for j in range(idx.shape[1]):
                lag_blocks = [X[:, :, idx[i, j]] for i in range(idx.shape[0])]
                feats_per_t.append(np.concatenate(lag_blocks, axis=1))

            F = np.stack(feats_per_t, axis=2)

            if lag_agg == "mean":
                return F.mean(axis=2)
            elif lag_agg == "stack_time":
                return F.reshape(n_trials, (n_neurons * len(lags)) * F.shape[2])
            else:
                raise ValueError(f"Unknown lag_agg='{lag_agg}'")

        raise ValueError(f"Unknown mode='{mode}'. Use 'mean', 'stack', or 'lagged'.")

    def make_timecourse_feature_tensor(data_single_sess, t_start, t_end, ref_window_key):
        X = np.asarray(data_single_sess)
        if X.ndim != 3:
            raise ValueError(f"Expected (trials, neurons, time), got {X.shape}")

        n_trials, n_neurons, n_time = X.shape

        if not (0 <= t_start < t_end <= n_time):
            raise ValueError(f"Invalid timecourse range [{t_start},{t_end}) for n_time={n_time}")

        if zscore_time:
            b0, b1 = baseline_window
            X = baseline_zscore(X, b0, b1, eps=baseline_eps)

        if mode == "mean":
            ts = np.arange(t_start, t_end, 1, dtype=int)
            X_feat_tc = X[:, :, ts]
            return X_feat_tc, ts

        if mode == "stack":
            w0, w1 = windows[ref_window_key]
            rel = np.arange(0, (w1 - w0), step, dtype=int)

            starts = np.arange(t_start, t_end, 1, dtype=int)
            valid = (starts + rel[-1] < n_time)
            starts = starts[valid]
            if len(starts) == 0:
                raise ValueError("No valid timepoints for stack timecourse (window too close to end).")

            feat_list = []
            for s in starts:
                idx = s + rel
                feat_s = X[:, :, idx].reshape(n_trials, n_neurons * len(rel))
                feat_list.append(feat_s)
            X_feat_tc = np.stack(feat_list, axis=2)
            return X_feat_tc, starts

        if mode == "lagged":
            lag_arr = np.array(lags, dtype=int)
            ts = np.arange(t_start, t_end, 1, dtype=int)
            idx = ts[None, :] + lag_arr[:, None]
            valid = np.all((idx >= 0) & (idx < n_time), axis=0)
            ts = ts[valid]
            idx = idx[:, valid]
            if idx.shape[1] == 0:
                raise ValueError("No valid timepoints for lagged timecourse with given lags.")

            feat_list = []
            for j in range(idx.shape[1]):
                lag_blocks = [X[:, :, idx[i, j]] for i in range(idx.shape[0])]
                feat_t = np.concatenate(lag_blocks, axis=1)
                feat_list.append(feat_t)
            X_feat_tc = np.stack(feat_list, axis=2)
            return X_feat_tc, ts

        raise ValueError(f"Unknown mode='{mode}'")

    z_seq_by_sess = {k: {} for k in axis_defs}
    z_trials_by_sess = {k: {} for k in axis_defs}
    z_tc_by_sess = {k: {} for k in axis_defs}
    z_tc_trials_by_sess = {k: {} for k in axis_defs}

    angle_by_sess = {}
    acc_by_sess = {k: {} for k in axis_defs}
    cv_acc_by_sess = {k: {} for k in axis_defs}
    cv_shuff_acc_by_sess = {k: {} for k in axis_defs}

    for session_key in tqdm(sessions_by_id, total=len(sessions_by_id), desc='Processing sessions'):
        one_session = data[session_key]

        for axis_name in axis_defs:
            z_seq_by_sess[axis_name][session_key] = {seq: [] for seq in seq_types}
            z_trials_by_sess[axis_name][session_key] = {seq: [] for seq in seq_types}
            z_tc_by_sess[axis_name][session_key] = {seq: [] for seq in seq_types}
            z_tc_trials_by_sess[axis_name][session_key] = {seq: [] for seq in seq_types}

        X_train = {k: [] for k in axis_defs}
        y_train = {k: [] for k in axis_defs}

        for seq_key in seq_types:
            data_single_sess = one_session[seq_key]
            num_trials = data_single_sess.shape[0]

            for axis_name, axis_def in axis_defs.items():
                t0, t1 = windows[axis_def['window']]
                X_tmp = make_temporal_X(data_single_sess, t0, t1)

                if axis_def['label'] == 'pitch':
                    y_tmp = start_freqs[seq_key] * np.ones(num_trials)
                else:
                    y_tmp = pattern_types[seq_key] * np.ones(num_trials)

                X_train[axis_name] = (
                    np.concatenate((X_train[axis_name], X_tmp), axis=0)
                    if len(X_train[axis_name]) > 0 else X_tmp
                )
                y_train[axis_name] = np.concatenate((y_train[axis_name], y_tmp))

        per_axis_pca = {ax: None for ax in axis_defs}
        if use_pca:
            if shared_pca:
                feat_dims = {ax: X_train[ax].shape[1] for ax in axis_defs}
                dims = list(feat_dims.values())
                if not all(d == dims[0] for d in dims):
                    raise ValueError(f"shared_pca=True requires same feature dimension across axes.")
                X_all = np.concatenate([X_train[ax] for ax in axis_defs], axis=0)
                session_pca = PCA(n_components=pca_n_components, whiten=pca_whiten).fit(X_all)
                for ax in axis_defs:
                    per_axis_pca[ax] = session_pca
            else:
                for ax in axis_defs:
                    per_axis_pca[ax] = PCA(n_components=pca_n_components, whiten=pca_whiten).fit(X_train[ax])

        axes = {}

        for axis_name in axis_defs:
            X = X_train[axis_name]
            y = y_train[axis_name]

            pca = per_axis_pca[axis_name]
            X_fit = pca.transform(X) if pca is not None else X

            lda = LinearDiscriminantAnalysis(solver=lda_solver, shrinkage=lda_shrinkage)
            lda.fit(X_fit, y)

            acc_by_sess[axis_name][session_key] = accuracy_score(y, lda.predict(X_fit))

            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            cv_acc = []
            cv_shuff_acc_list = []

            rng = np.random.default_rng(12345)

            for tr, te in skf.split(X, y):
                lda_fold = LinearDiscriminantAnalysis(solver=lda_solver, shrinkage=lda_shrinkage)
                lda_fold.fit(X[tr], y[tr])
                cv_acc.append(accuracy_score(y[te], lda_fold.predict(X[te])))

                fold_shuff_scores = []
                for _ in range(n_shuff):
                    y_tr_shuff = y[tr].copy()
                    rng.shuffle(y_tr_shuff)
                    lda_shuff = LinearDiscriminantAnalysis(solver=lda_solver, shrinkage=lda_shrinkage)
                    lda_shuff.fit(X[tr], y_tr_shuff)
                    fold_shuff_scores.append(accuracy_score(y[te], lda_shuff.predict(X[te])))

                cv_shuff_acc_list.append(np.mean(fold_shuff_scores))

            cv_acc_by_sess[axis_name][session_key] = float(np.mean(cv_acc))
            cv_shuff_acc_by_sess[axis_name][session_key] = float(np.mean(cv_shuff_acc_list))

            w = lda.coef_[0].copy()
            b = float(lda.intercept_[0])

            norm = np.linalg.norm(w)
            if norm == 0:
                norm = 1.0
            w = w / norm
            b = b / norm

            axes[axis_name] = {'w': w, 'b': b, 'lda': lda}

        angle_dict = {}
        for a1 in axes:
            for a2 in axes:
                cos_theta = np.dot(axes[a1]['w'], axes[a2]['w'])
                angle_dict[(a1, a2)] = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        angle_by_sess[session_key] = angle_dict

        for axis_name, axis_def in axis_defs.items():
            w = axes[axis_name]['w']
            b = axes[axis_name]['b']
            pca = per_axis_pca[axis_name]

            for seq_key in seq_types:
                data_single_sess = one_session[seq_key]
                t0, t1 = windows[axis_def['window']]
                X_trial = make_temporal_X(data_single_sess, t0, t1)
                if pca is not None:
                    X_trial = pca.transform(X_trial)
                z_trial = X_trial @ w + b
                z_trials_by_sess[axis_name][session_key][seq_key] = z_trial

        for axis_name in axis_defs:
            for seq_key in seq_types:
                z_seq_by_sess[axis_name][session_key][seq_key] = (
                    z_trials_by_sess[axis_name][session_key][seq_key].mean()
                )

        any_seq = seq_types[0]
        n_time = one_session[any_seq].shape[2]

        for axis_name, axis_def in axis_defs.items():
            w = axes[axis_name]['w']
            b = axes[axis_name]['b']
            pca = per_axis_pca[axis_name]

            if timecourse_t_range == "window":
                t_start, t_end = windows[axis_def['window']]
            else:
                t_start, t_end = 0, n_time

            for seq_key in seq_types:
                data_single_sess = one_session[seq_key]
                X_tc, t_idx = make_timecourse_feature_tensor(
                    data_single_sess=data_single_sess,
                    t_start=t_start,
                    t_end=t_end,
                    ref_window_key=axis_def['window']
                )

                if pca is not None:
                    n_trials, n_feat, n_t = X_tc.shape
                    X_flat = np.transpose(X_tc, (0, 2, 1)).reshape(n_trials * n_t, n_feat)
                    X_flat = pca.transform(X_flat)
                    n_feat2 = X_flat.shape[1]
                    X_tc = X_flat.reshape(n_trials, n_t, n_feat2).transpose(0, 2, 1)

                z_trials_tc = np.tensordot(X_tc, w, axes=([1], [0])) + b
                z_mean_tc = z_trials_tc.mean(axis=0)

                z_tc_trials_by_sess[axis_name][session_key][seq_key] = z_trials_tc
                z_tc_by_sess[axis_name][session_key][seq_key] = z_mean_tc

    return (
        z_seq_by_sess,
        angle_by_sess,
        acc_by_sess,
        cv_acc_by_sess,
        cv_shuff_acc_by_sess,
        z_trials_by_sess,
        z_tc_by_sess,
        z_tc_trials_by_sess
    )


def plot_LDA_angle_matrix(angle_by_sess, axis_names, bins=20):
    n_axes = len(axis_names)
    fig, axs = plt.subplots(n_axes, n_axes, figsize=(2 * n_axes, 2 * n_axes), sharex=True, sharey=True)

    for i, ax1 in enumerate(axis_names):
        for j, ax2 in enumerate(axis_names):
            ax = axs[i, j]
            if i > j:
                ax.axis('off')
                continue

            if ax1 == ax2:
                ax.hist([0], bins=bins, alpha=0.7)
                ax.axvline(0, color='black', linewidth=2)
                ax.set_xlim(-50, 180)
                ax.set_ylim(0, 45)
            else:
                angles = [angle_by_sess[session_key][(ax1, ax2)] for session_key in angle_by_sess]
                ax.hist(angles, bins=bins, alpha=0.7)
                mean_angle = np.mean(angles)
                ax.axvline(mean_angle, color='blue', linestyle='dashed', linewidth=1, label=f"Mean = {mean_angle:.4f}°")
                ax.axvline(90, color='red', linestyle='dashed', linewidth=1)
                ax.set_xlim(-50, 180)
                ax.set_ylim(0, 45)
                ax.legend(fontsize=8, frameon=False)

            if i == 0:
                ax.set_title(ax2)
            if i == n_axes - 1:
                ax.set_xlabel("Angle (deg)")

    for i, ax1 in enumerate(axis_names):
        y = 1 - (i + 0.5) / n_axes
        fig.text(-0.02, y, ax1, va='center', ha='right', fontsize=11)

    fig.suptitle("Distribution of LDA axis angles across sessions", y=1.02)
    fig.tight_layout(rect=[0.05, 0, 1, 0.96])
    fig.savefig("figs/lda_angle_matrix.svg", bbox_inches="tight", dpi=600)
    fig.savefig("figs/lda_angle_matrix.pdf", bbox_inches="tight", dpi=600)
    plt.close(fig)


def plot_LDA_accuracy_histograms(acc_by_sess, cv_acc_by_sess, bins=10):
    axis_names = list(acc_by_sess.keys())
    n_axes = len(axis_names)
    fig, axs = plt.subplots(1, n_axes, figsize=(4 * n_axes, 3), sharey=True)

    if n_axes == 1:
        axs = [axs]

    for ax, axis_name in zip(axs, axis_names):
        acc_vals = np.array(list(acc_by_sess[axis_name].values()))
        cv_vals = np.array(list(cv_acc_by_sess[axis_name].values()))

        ax.hist(acc_vals, bins=bins, alpha=0.5, color='blue', label='train acc')
        ax.axvline(acc_vals.mean(), color='blue', linestyle='dashed', linewidth=1, label=f'Mean Train acc = {acc_vals.mean():.4f}')
        ax.hist(cv_vals, bins=bins, alpha=0.5, color='orange', label=f'5-fold CV')
        ax.axvline(cv_vals.mean(), color='orange', linestyle='dashed', linewidth=1, label=f'Mean CV acc = {cv_vals.mean():.4f}')

        ax.set_title(axis_name)
        ax.set_xlabel('Accuracy')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    axs[0].set_ylabel('Number of sessions')
    fig.suptitle('Distribution of LDA accuracy across sessions', y=1.05)
    fig.tight_layout()
    fig.savefig("figs/lda_accuracy_histograms.svg", bbox_inches="tight", dpi=600)
    fig.savefig("figs/lda_accuracy_histograms.pdf", bbox_inches="tight", dpi=600)


def compute_rule_axis_alignment(
        z_tc_trials_by_sess,
        sessions_by_id,
        axis_name="pattern4",
        abba_seqs=("ABBA0", "ABBA1"),
        abcd_seqs=("ABCD0", "ABCD1"),
        align_positive=True,
        peak_window=None,
):
    """
    Compute per-session rule axis alignment timecourse: shape (n_sessions, n_time).
    Alignment at each timepoint is mean(z_ABBA) - mean(z_ABCD).
    If `align_positive=True`, flips a session's sign if its mean alignment
    over `peak_window` (or overall) is negative so axis direction matches rule emergence.
    """
    alignments = []

    for sess in sessions_by_id:
        sess_tc = z_tc_trials_by_sess[axis_name][sess]

        abba_trials = np.concatenate([np.asarray(sess_tc[s]) for s in abba_seqs], axis=0)
        abcd_trials = np.concatenate([np.asarray(sess_tc[s]) for s in abcd_seqs], axis=0)

        # Shape: (n_time,)
        align_tc = abba_trials.mean(axis=0) - abcd_trials.mean(axis=0)

        if align_positive:
            ref_slice = slice(None) if peak_window is None else slice(peak_window[0], peak_window[1])
            if np.nanmean(align_tc[ref_slice]) < 0:
                align_tc = -align_tc

        alignments.append(align_tc)

    return np.asarray(alignments, dtype=float)


def plot_rule_axis_emergence(
        x_ser,
        alignment_by_sess,
        pip_starts=(0.0, 0.25, 0.5, 0.75),
        pip_duration=0.15,
        star_mode="peak",
        alpha_p=0.05,
        save_path=None,
):
    """
    Reproduce the 'Rule axis emergence' timecourse plot using Figure/Axes methods.
    `star_mode`: 'peak' places a red star at the maximum of the mean curve;
                 'sig' places red stars at timepoints significant vs. 0 (one-sample t-test).
    """
    mean_tc = np.nanmean(alignment_by_sess, axis=0)
    n_sess = alignment_by_sess.shape[0]
    sem_tc = np.nanstd(alignment_by_sess, axis=0, ddof=1) / np.sqrt(n_sess)

    fig, ax = plt.subplots(figsize=(2,2))

    # Shaded stimulus pip intervals
    for t0 in pip_starts:
        ax.axvspan(t0, t0 + pip_duration, color="#efefef", lw=0, zorder=0)

    # Reference lines at x = 0 and y = 0
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.1, zorder=1)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.5, zorder=2)

    # SEM ribbon and mean alignment curve
    ax.fill_between(
        x_ser,
        mean_tc - sem_tc,
        mean_tc + sem_tc,
        color="lightgray",
        alpha=0.7,
        linewidth=0,
        zorder=2,
    )
    ax.plot(x_ser, mean_tc, color="k", linewidth=2.0, zorder=3)

    ax.set_ylabel("Axis alignment")
    ax.set_xlim(-0.35, 1.45)
    # ax.set_ylim(-0.05, max(0.85, star_y + 0.07))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    # ax.set_yticks([0.00, 0.25, 0.50, 0.75])
    # ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75"], fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", direction="in", length=5, width=1.1, top=False, right=False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)

    fig.show()
    return fig, ax


def compute_axis_alignment_timecourse(
        z_tc_trials_by_sess,
        sessions_by_id,
        axis_name,
        pos_seqs,
        neg_seqs,
        align_positive=True,
        peak_window=None,
):
    """
    Compute per-session alignment timecourse: shape (n_sessions, n_time).
    At each timepoint, alignment is mean(z[pos_seqs]) - mean(z[neg_seqs]).
    If `align_positive=True`, flips a session's sign when its mean alignment
    over `peak_window` (or full timecourse) is negative.
    """
    alignments = []

    for sess in sessions_by_id:
        sess_tc = z_tc_trials_by_sess[axis_name][sess]

        pos_trials = np.concatenate([np.asarray(sess_tc[s]) for s in pos_seqs], axis=0)
        neg_trials = np.concatenate([np.asarray(sess_tc[s]) for s in neg_seqs], axis=0)

        align_tc = pos_trials.mean(axis=0) - neg_trials.mean(axis=0)

        if align_positive:
            ref_slice = slice(None) if peak_window is None else slice(peak_window[0], peak_window[1])
            if np.nanmean(align_tc[ref_slice]) < 0:
                align_tc = -align_tc

        alignments.append(align_tc)

    return np.asarray(alignments, dtype=float)


def plot_axis_emergence(
        x_ser,
        alignment_by_sess,
        title="Axis emergence",
        ylabel="Axis alignment",
        pip_starts=(0.0, 0.25, 0.5, 0.75),
        pip_duration=0.15,
        star_mode="none",
        alpha_p=0.05,
        xlim=(-0.35, 1.45),
        ylim=None,
        yticks=None,
        figsize=(2.2, 2.2),
        save_path=None,
):
    """
    Plot mean +/- SEM axis alignment timecourse using Figure and Axes methods.
    `star_mode`: 'none', 'peak' (red star at peak of mean curve), or
                 'sig' (red stars where one-sample t-test vs. 0 < alpha_p).
    """
    mean_tc = np.nanmean(alignment_by_sess, axis=0)
    n_sess = alignment_by_sess.shape[0]
    sem_tc = np.nanstd(alignment_by_sess, axis=0, ddof=1) / np.sqrt(n_sess)

    fig, ax = plt.subplots(figsize=figsize)

    for t0 in pip_starts:
        ax.axvspan(t0, t0 + pip_duration, color="#efefef", lw=0.0, zorder=0)

    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.1, zorder=1)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.5, zorder=2)

    ax.fill_between(
        x_ser,
        mean_tc - sem_tc,
        mean_tc + sem_tc,
        color="lightgray",
        alpha=0.7,
        linewidth=0.0,
        zorder=2,
    )
    ax.plot(x_ser, mean_tc, color="k", linewidth=2.0, zorder=3)

    if star_mode in ("peak", "sig"):
        y_top = np.nanmax(mean_tc + sem_tc)
        star_y = y_top + 0.08 * max(abs(y_top), 0.5)

        if star_mode == "peak":
            peak_idx = int(np.nanargmax(mean_tc))
            ax.scatter(
                [x_ser[peak_idx]],
                [star_y],
                marker="*",
                s=85,
                facecolor="#d61f1f",
                edgecolor="k",
                linewidth=0.6,
                zorder=5,
            )
        elif star_mode == "sig":
            _, p_vals = ttest_1samp(alignment_by_sess, popmean=0.0, axis=0, nan_policy="omit")
            sig_mask = p_vals < alpha_p
            if np.any(sig_mask):
                ax.scatter(
                    x_ser[sig_mask],
                    np.full(np.sum(sig_mask), star_y),
                    marker="*",
                    s=70,
                    facecolor="#d61f1f",
                    edgecolor="k",
                    linewidth=0.6,
                    zorder=5,
                )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", direction="in", length=5, width=1.1, top=False, right=False)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=600)

    return fig, ax


def window_index_bounds(x_ser, t_start, t_end):
    idx = np.where((x_ser >= t_start) & (x_ser <= t_end))[0]
    if len(idx) == 0:
        return None
    return int(idx[0]), int(idx[-1]) + 1
