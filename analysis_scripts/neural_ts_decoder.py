from types import SimpleNamespace
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def decode_responses(xs, ys, dec_kwargs=None):
    """
    Decode neural responses `xs` (n_trials, n_features) for labels `ys` (n_trials,)
    across `n_runs` balanced subsamples using 5-fold CV and LogisticRegression.
    """
    kwargs = {} if dec_kwargs is None else dict(dec_kwargs)
    n_runs = kwargs.get("n_runs", 100)
    cv_folds = kwargs.get("cv_folds", 5)
    seed = kwargs.get("seed", 42)
    time_bin = kwargs.get("time_bin", 0)
    max_iter = kwargs.get("max_iter", 1000)
    c_val = kwargs.get("C", 1.0)

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    classes, counts = np.unique(ys, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes for decoding.")

    min_count = counts.min()
    if min_count < cv_folds:
        raise ValueError(
            f"Smallest class has {min_count} samples, fewer than cv_folds={cv_folds}."
        )

    # Same deterministic RNG progression for a given time_bin across runs
    rng = np.random.default_rng(seed + int(time_bin))
    run_accuracies = np.empty(n_runs, dtype=float)

    for run_idx in range(n_runs):
        selected_indices = []
        for cls in classes:
            cls_idx = np.flatnonzero(ys == cls)
            chosen = rng.choice(cls_idx, size=min_count, replace=False)
            selected_indices.append(chosen)

        bal_idx = np.concatenate(selected_indices)
        # Shuffle balanced trials before splitting into folds
        bal_idx = rng.permutation(bal_idx)

        x_bal = xs[bal_idx]
        y_bal = ys[bal_idx]

        fold_seed = int(rng.integers(0, 2**31 - 1))
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=fold_seed)

        clf = make_pipeline(
            StandardScaler(),
            # LogisticRegression(C=c_val, max_iter=max_iter, solver="lbfgs"),
            LogisticRegression(class_weight='balanced', max_iter=10000, solver='newton-cg',
                               penalty='l2', C=0.1)
            ,
        )

        scores = cross_val_score(clf, x_bal, y_bal, cv=cv, scoring="accuracy")
        run_accuracies[run_idx] = scores.mean()

    return {
        "data": SimpleNamespace(
            accuracy=run_accuracies,
            mean_accuracy=float(run_accuracies.mean()),
        )
    }