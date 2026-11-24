import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s
from quantities.constants import alpha
from scipy.stats import linregress, ttest_ind, ttest_rel
import statsmodels.formula.api as smf
from tqdm import tqdm

from pupil_analysis_funcs import add_name_to_response_dfs, run_pupil_cond_analysis
from plot_funcs import plot_shaded_error_ts, format_axis
from xdetectioncore.behaviour import get_all_cond_filts, load_aggregate_td_df, get_n_since_last
from xdetectioncore.io_utils import load_response_dfs

# =====================================================================
# 1. CORE DATA WRANGLING INTERFACES
# =====================================================================

def prepare_continuous_time_series(data_dict, feature_cols_to_keep=None):
    """
    Stacks wide condition dataframes into a long format while completely
    preserving the original MultiIndex and tracking the time series.
    """
    all_data = []

    for stim_name, df_stim in data_dict.items():
        df_temp = df_stim.copy()

        # 1. Filter out unwanted features from the MultiIndex if specified
        if feature_cols_to_keep is not None:
            # Drop index levels that are NOT in your keep list
            levels_to_drop = [name for name in df_temp.index.names if
                              name not in feature_cols_to_keep and name is not None]
            if levels_to_drop:
                # Dropping levels collapses rows; you may want to aggregate here if duplicates occur,
                # but droplevel handles structural reduction.
                df_temp = df_temp.droplevel(levels_to_drop)

        # 2. Pivot columns (Time) into the MultiIndex rows using .stack()
        # This keeps your original index perfectly preserved and appends 'Time' as the final level.
        series_long = df_temp.stack(future_stack=True)

        # 3. Give names to the newly created index level and the value series
        series_long.index = series_long.index.set_names('Time', level=-1)
        series_long.name = 'Response'

        # Convert back to DataFrame while keeping everything in the Index
        df_long = series_long.to_frame()

        # 4. Optional: Clean up the 'Time' level to numeric format if needed
        # Since it's a MultiIndex level, we modify it via set_levels
        time_level_idx = df_long.index.names.index('Time')
        df_long.index = df_long.index.set_levels(
            pd.to_numeric(df_long.index.levels[time_level_idx]),
            level='Time'
        )
        df_long['Time'] = df_long.index.get_level_values('Time')
        df_long.droplevel('Time')

        all_data.append(df_long)

    # Combine all dataframes along the index axis
    df_master = pd.concat(all_data, axis=0).dropna()

    # Return the master long dataframe (where all features + Time are in the MultiIndex)
    return df_master

def compute_feve(df, target_feature, joint_features, metric):
    """
    Computes the Fraction of Explainable Variance Explained (FEVE).

    Formula: FEVE = Deviance_Feature / Deviance_Full_Model

    Parameters:
    - df: The dataframe containing the data (must have the target and joint features).
    - target_feature: String, the specific feature to test (e.g., 'local_rate').
    - joint_features: List of strings, all features in the 'full' model (e.g., ['local_rate', 'n_since_last']).
    - metric: String, the dependent variable to predict (e.g., 'Peak_Amplitude').

    Returns:
    - feve_ratio: float, the proportion of explainable variance.
    """
    # 1. Prepare data for the Full Model (the 'ceiling')
    X_full = df[joint_features].values
    y = df[metric].values

    # Fit full model
    # Note: Using standard spline formulation; adjust based on your specific GAM setup
    gam_full = LinearGAM(n_splines=15).gridsearch(X_full, y, progress=False)
    dev_full = gam_full.statistics_['pseudo_r2']['explained_deviance']

    # 2. Prepare data for the Feature-Specific Model
    X_feat = df[[target_feature]].values

    # Fit single-feature model
    gam_feat = LinearGAM(n_splines=15).gridsearch(X_feat, y, progress=False)
    dev_feat = gam_feat.statistics_['pseudo_r2']['explained_deviance']

    # 3. Calculate Ratio
    # We add a tiny epsilon to avoid division by zero if dev_full is 0
    feve_ratio = dev_feat / (dev_full + 1e-9)

    return feve_ratio, dev_feat, dev_full

# =====================================================================

print("Loading raw files and building MultiIndex metadata indices...")
session_topology_paths = [
        # ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_sept23.csv'),
        # ceph_dir / posix_from_win(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_2406.csv'),
       Path(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_opto_2503.csv'),
       Path(r'X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2504.csv'),
    ]

# Concatenate session topology dataframes
session_topology_dfs = [pd.read_csv(sess_topology_path) for sess_topology_path in session_topology_paths]
session_topology_df = pd.concat(session_topology_dfs, names=['cohort']).reset_index()
all_sess_info = session_topology_df.query('sess_order=="main"')

all_td_df = load_aggregate_td_df(all_sess_info,Path('H:\\'))
# all_td_df = pd.read_hdf("patt_resp_td_df_ephys_2504_opto_2503.h5", )
all_td_df = all_td_df[all_td_df.index.to_frame().duplicated() == False]
all_td_df = all_td_df.sort_index()
patt_trials = all_td_df.loc[all_td_df['Tone_Position']==0]


rate_win_size = 10
for idx, patt_t in tqdm(patt_trials.iterrows(), total=len(patt_trials),desc='Adding local rate'):
    if idx[-1]<=rate_win_size:
        all_td_df.loc[idx,'local_rate'] = 0
    else:
        # sess_td = all_td_df.xs(key=idx[-2],level='sess')
        all_td_df.loc[idx,'local_rate']=all_td_df.loc[idx[0],idx[1],idx[2],idx[-1]-rate_win_size:idx[-1]]['Tone_Position'].sum()/rate_win_size

for sessname, sess_df in all_td_df.groupby('sess'):
    sess_patt_trials = sess_df[sess_df['Tone_Position']==0]
    patt_times_as_int = (sess_patt_trials['ToneTime_dt']-sess_df['Trial_Start_dt'].iloc[0]).dt.total_seconds()
    all_td_df.loc[patt_times_as_int.index,'ToneTime_diff'] = patt_times_as_int.diff()

# all_td_df['n_since_last'] = get_n_since_last(all_td_df,'Tone_Position',0)
patt_hist_df = all_td_df[['local_rate', 'n_since_last','ToneTime_diff']].copy()
patt_hist_df.index.rename({'trial_num': 'trial'}, inplace=True)

A_by_cond_hdf_path = "pattern_resps_opto.h5"
A_by_cond = load_response_dfs(A_by_cond_hdf_path)

for k,v in A_by_cond.items():
    A_by_cond[k] = v.drop_duplicates()


with open(r'H:\ephys\sesslist_by_cond.json', 'r') as f:
    sesslist_by_cond = json.load(f)

for cond, df in A_by_cond.items():
    if 'local_rate' in df.index.names:
        continue

    # 1. Extract the unique sess and trial combinations as a DataFrame
    pairs_df = df.index.to_frame()[["sess", "trial"]].drop_duplicates().copy()

    df = df.copy()
    if len(df.index.get_level_values('sess')[0].split('_')) == 3:
        idx_df = df.index.to_frame()
        idx_df['sess'] = idx_df['sess'].str.rsplit('_', n=1).apply(lambda x: x[0])
        df.index = pd.MultiIndex.from_frame(idx_df)
    # df = df.droplevel(level='time')

    # 1. Zip the 'B' and 'C' levels of patt_hist_df into a list of tuples
    df1_bc_values = list(
        zip(
            patt_hist_df.index.get_level_values("sess"),
            patt_hist_df.index.get_level_values("trial"),
        )
    )

    # 2. Zip the 'B' and 'C' levels of df into a set for fast O(1) lookups
    df2_bc_values = set(
        zip(df.index.get_level_values("sess"),
            df.index.get_level_values("trial"))
    )

    # 3. Build a boolean mask checking which pairs match
    mask = [pair in df2_bc_values for pair in df1_bc_values]

    # 4. Filter patt_hist_df safely
    result = patt_hist_df[mask]

    # 1. Pick the column from 'result' that you want to add to the index
    columns_to_add = ['local_rate','n_since_last','ToneTime_diff']
    for column_to_add in columns_to_add:

        # 2. Extract the column and the shared (sess, trial) keys from 'result'
        result_flat = result[[column_to_add]].reset_index()

        # 3. Drop duplicate (sess, trial) combinations to prevent mapping inflation
        result_flat = result_flat.drop_duplicates(subset=["sess", "trial"])

        # 4. Set 'sess' and 'trial' as a temporary mapping index
        mapping_series = result_flat.set_index(["sess", "trial"])[column_to_add]

        # 5. Extract df's 'sess' and 'trial' levels into a clean MultiIndex structure
        df_keys = pd.MultiIndex.from_arrays(
            [df.index.get_level_values("sess"), df.index.get_level_values("trial"),
             df.index.get_level_values("time")],
            names=["sess", "trial",'time'],
        )

        # 6. Align the mapped values to df's rows perfectly
        mapped_values = mapping_series.reindex(df_keys).values

        # 7. Safely append this as a new level to df's index
        df = df.set_index(pd.Index(mapped_values, name=column_to_add), append=True)

    A_by_cond[cond] = df

# # Standardize extraction labels to avoid key crashes
A_by_cond['pattern_rel_rec'] = A_by_cond['pattern'].copy()
data_dict = {'pattern_rel_rec': A_by_cond['pattern_rel_rec']}

# =====================================================================
# PUPIL CONDITION ANALYSIS
# =====================================================================

print("Isolating isolated event rate profiles...")
add_name_to_response_dfs(A_by_cond)
data_dict['pattern'] = A_by_cond['pattern'].copy()

for n_last in [1,2]:
    post_patt_reps = {
        'rare_no_pre_patt': data_dict['pattern'].query('n_since_last==@n_last & local_rate >= 0.8'),
        'frequent_no_pre_patt': data_dict['pattern'].query('n_since_last==@n_last & local_rate <=0.2')
    }

    post_patt_sess_2_keep = np.intersect1d(*[
        resps.groupby(['sess', 'name']).filter(lambda x: len(x) >= 0).index.get_level_values('sess').unique()
        for resps in post_patt_reps.values()
    ])

    fig_pupil, ax_pupil = plt.subplots(figsize=(7, 4.5))
    color_map = {'rare': '#00cbfcff', 'frequent': '#1530e9ff'}

    for cond, resps in post_patt_reps.items():
        resps_filtered = resps.loc[resps.index.get_level_values('sess').isin(post_patt_sess_2_keep)]

        resps_filtered = resps_filtered

        mean_series = resps_filtered.groupby(['sess', 'name']).mean().mean()
        sem_series = resps_filtered.groupby(['sess', 'name']).mean().sem()

        cond_key = cond.split('_')[0]
        ax_pupil.plot(mean_series, label=cond, c=color_map[cond_key])
        plot_shaded_error_ts(ax_pupil, resps_filtered.columns, mean_series, sem_series, fc=color_map[cond_key], alpha=0.1)

    format_axis(
        ax_pupil,
        xlim=(-0.5, 2.5),
        vlines=[0],
        vspan=[[t, t + 0.15] for t in np.arange(0, 1, 0.25)],
        title='Event Rate Effect (t-1: patt)',
        xlabel='Time (s)',
        ylabel=r'$\Delta$ Pupil Size'
    )
    ax_pupil.legend()
    fig_pupil.show()
    fig_pupil.savefig(f'Pupil_Rate_Effect_n_since_last=={n_last}.pdf')

    rare_freq_line_kwargs = {
        'rare': {'c': '#00cbfcff', 'ls': '-'},
        'frequent': {'c': '#1530e9ff', 'ls': '-'}
    }
    analysis = run_pupil_cond_analysis(
                    by_cond_dict={c:e for c,e in zip(['rare','frequent'],post_patt_reps.values())},
                    sess_list=post_patt_sess_2_keep,
                    conditions=['rare','frequent'],
                    figdir=Path(''),
                    line_kwargs=rare_freq_line_kwargs,
                    boxplot_kwargs=None,
                    window_by_stim=(1, 2),
                    smoothing_window=5,
                    max_window_kwargs={'max_func': 'max'},
                    group_name='sess',
                    permutation_test=False,
                    n_permutations=100,
                    cluster_groupname=['sess'],
                    cluster_comps=[['frequent','rare']],
                    stats_dir=Path(''),
                    # fig_savename=f'rarefreq_early_late_{opto_state}_{batch_name}_ts.pdf',
                    tex_name=f'fill',
                    ylabel="Δ pupil size",
                    figsize=(1.6, 2),
                    ylim_ts=(-0.15, 0.45),
                    # ylim_maxdiff=(-0.25, 0.25),
                    inset_max_diff=False
                )


# =====================================================================
# 1. STAGE 0: VECTORIZED TEMPORAL DETRENDING
# =====================================================================
print("Regressing out Time from long-form response series via GAM...")

data_dict['pattern_bs'] = data_dict['pattern'].subtract(data_dict['pattern'].loc[:,-1:0].mean(axis=1),axis=0).copy()
df_tidy = prepare_continuous_time_series({'d':data_dict['pattern_bs']},
                                                       ['sess','name','trial','local_rate',
                                                        'n_since_last','ToneTime_diff'])
df_tidy = df_tidy.droplevel('Time')
for col in ['local_rate', 'n_since_last','ToneTime_diff']:
    df_tidy[col] = df_tidy.index.get_level_values(col)
    df_tidy = df_tidy.droplevel(col)

X_time = df_tidy[['Time']].values
y_resp = df_tidy['Response'].values

# Fit spline matrix across all timestamps to isolate baseline pupil dynamics
time_gam = LinearGAM(s(0, n_splines=25)).gridsearch(X_time, y_resp)
df_tidy['Response_Residual'] = y_resp - time_gam.predict(X_time)

# Format
_df_dict = {'df':df_tidy}
add_name_to_response_dfs(_df_dict)
df_tidy = _df_dict['df']

# =====================================================================
# 2. STAGE 1: COMPUTE ALL TRIAL METRICS SEPARATED BY ANIMAL ('name')
# =====================================================================
print("Extracting multi-metric trial matrix grouped by individual animals...")

# Isolate post-stimulus windows
df_post = df_tidy[np.all([df_tidy['Time'] > 1.0,df_tidy['Time']<1.5],axis=0)].copy()
unique_names = df_post.index.get_level_values('name').unique()

# def _extract_all_metrics(group,resp_colname='Response_Residual'):
def _extract_all_metrics(group,resp_colname='Response'):
    """
    Computes ALL 9 physiological metrics for a single trial block
    and returns them unified under a single series structure.
    """
    group = group.sort_values(by='Time')
    t = group['Time'].values
    r = group[resp_colname].values

    # Base Metrics
    mean_resp = np.mean(r)
    idx_max = np.argmax(r)
    peak_amp = r[idx_max]
    auc_val = np.trapz(np.clip(r, 0, None), t)

    # Advanced Metrics
    velocity = np.gradient(r, t)
    # max_velocity = np.max(velocity)

    trough_amp = np.min(r)
    excursion = peak_amp - trough_amp

    half_max = peak_amp / 2.0
    above_half_indices = np.where(r >= half_max)[0]
    if len(above_half_indices) > 1:
        fwhm_duration = t[above_half_indices[-1]] - t[above_half_indices[0]]
    else:
        fwhm_duration = 0.0

    rms_val = np.sqrt(np.mean(r ** 2))
    mean_abs_area = np.mean(np.abs(r))

    return pd.Series({
        'Mean_Response':       mean_resp,
        'Peak_Amplitude':      peak_amp,
        'Latency_to_Peak':     t[idx_max],
        'RMS_Energy':          rms_val,
        'AUC_Above_Zero':      auc_val,
        'Peak_to_Trough':      excursion,
        'FWHM_Duration':       fwhm_duration,
        'Mean_Absolute_Area':  mean_abs_area
    })

print("Extracting multi-metric trial matrix grouped by 'name'...")

all_subjects_list = []
for name_id in unique_names:
    df_sub = df_post.xs(name_id, level='name')
    trial_groups = df_sub.groupby(level=['sess', 'trial'])

    # Compute everything inside our single unified function
    sub_metrics_df = trial_groups.apply(_extract_all_metrics, include_groups=False)

    # Grab independent features safely matching the multiindex structure
    sub_metrics_df['local_rate'] = trial_groups['local_rate'].first()
    sub_metrics_df['n_since_last'] = trial_groups['n_since_last'].first()
    sub_metrics_df['ToneTime_diff'] = trial_groups['ToneTime_diff'].first()
    sub_metrics_df['name'] = name_id

    all_subjects_list.append(sub_metrics_df)

master_trial_df = pd.concat(all_subjects_list).reset_index()
n_since_last_max = master_trial_df['n_since_last'].quantile([0.975]).values[0]
# master_trial_df = master_trial_df[master_trial_df['n_since_last'] <= n_since_last_max]
# n_since_last = master_trial_df['n_since_last'].values
# n_since_last[n_since_last>n_since_last_max] = n_since_last_max
# master_trial_df['n_since_last'] = n_since_last
for f in ['local_rate', 'n_since_last','ToneTime_diff']:
    master_trial_df[f'{f}_normed'] = (master_trial_df[f]-master_trial_df[f].mean())/master_trial_df[f].std()


print("\nFitting Independent GAMs Per Animal & Plotting GAM Fits...")

feature_names = ["local_rate_normed", "n_since_last", "ToneTime_diff"]
pd_by_feat = {}

for feature_name in feature_names:
    for metric in['Mean_Response']:
        # Drop missing records for the current feature and metric
        df_gam_input = master_trial_df.dropna(subset=[metric, feature_name, "name"]).copy()

        animal_names = []
        animal_deviances = []
        animal_curves = []

        # Common grid for evaluating and aggregating partial dependencies across animals
        feat_range = df_gam_input[feature_name].quantile([0, 0.95]).values
        X_grid = np.linspace(feat_range[0], feat_range[1], 100).reshape(-1, 1)

        for animal, df_animal in df_gam_input.groupby("name"):
            if len(df_animal) < 200:
                continue

            X_animal = df_animal[[feature_name]].values
            y_animal = df_animal[metric].values

            try:
                n_splines = min(15, len(df_animal) // 4)
                gam_animal = LinearGAM(s(0, n_splines=n_splines)).gridsearch(
                    X_animal, y_animal, progress=False
                )

                dev_explained = gam_animal.statistics_["pseudo_r2"]["explained_deviance"]
                p_dep = gam_animal.partial_dependence(term=0, X=X_grid)
                p_dep_centered = p_dep - np.mean(p_dep)

                animal_names.append(animal)
                animal_deviances.append(dev_explained)
                animal_curves.append(p_dep_centered)
                pd_by_feat[feature_name] = animal_deviances

            except Exception:
                continue

        if not animal_curves:
            continue

        animal_deviances = np.array(animal_deviances)
        animal_curves = np.array(animal_curves)  # (n_animals, 100)

        mean_deviance = np.mean(animal_deviances) * 100
        sem_deviance = (np.std(animal_deviances) / np.sqrt(len(animal_deviances))) * 100

        mean_curve = np.mean(animal_curves, axis=0)
        sem_curve = np.std(animal_curves, axis=0) / np.sqrt(len(animal_curves))

        gam_fit_plot, ax_gam = plt.subplots(figsize=[6, 4])

        ax_gam.plot(
            X_grid[:, 0],
            mean_curve,
            color="#2ca02c",
            linewidth=2.5,
            label=f"Population Mean (N={len(animal_names)})",
        )
        ax_gam.fill_between(
            X_grid[:, 0],
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color="#2ca02c",
            alpha=0.15,
            label="Across-Subject SEM",
        )

        ax_gam.set_title(
            f"Aggregated Effect of {feature_name} ({metric})",
            fontsize=11,
            fontweight="bold",
        )
        ax_gam.set_xlabel(f"{feature_name}")
        ax_gam.set_ylabel("Centered Partial Effect Shift")
        ax_gam.grid(True, linestyle=":", alpha=0.5)
        ax_gam.legend(frameon=False)
        ax_gam.spines[["top", "right"]].set_visible(False)

        gam_fit_plot.show()
        gam_fit_plot.savefig(f"gam_fit_{feature_name}_{metric}.pdf", bbox_inches="tight")

        print("\n============================================================")
        print(f" AGGREGATED INDEPENDENT MODEL METRICS: {metric}")
        print("============================================================")
        print(f"Number of Animals Successfully Modeled: {len(animal_names)}")
        print(
            f"Mean Variance Explained Across Animals: {mean_deviance:.2f}% ± {sem_deviance:.2f}% (SEM)"
        )
        print("\nIndividual Subject Breakdown:")
        for name, dev in zip(animal_names, animal_deviances):
            print(f"  - Animal {name}: {dev * 100:.2f}% Deviance Explained")
        print("============================================================")

##################################################



results_by_feature = []
for feature in feature_names:
    results_summary = []
    for name, group in tqdm(master_trial_df.groupby(['sess','name']),
                            total=len(master_trial_df['sess'].unique()),
                            desc=f"Computing FEVE features for {feature}"):
        if len(group) < 50:
            continue

        feve, dev_f, dev_full = compute_feve(
            group.dropna(),
            target_feature=feature,
            joint_features=feature_names,
            metric='Peak_Amplitude',
        )
        results_summary.append({'name': name, f'FEVE_{feature}': feve,
                                f'dev_f_{feature}': dev_f, f'dev_full_{feature}': dev_full})

    # Convert to DataFrame for easy reporting
    df_results = pd.DataFrame(results_summary)
    df_results.set_index('name', inplace=True)
    print(f"Population Mean FEVE for {feature}: {df_results[f'FEVE_{feature}'].mean():.3f}")
    results_by_feature.append(df_results)
feve_df = pd.concat(results_by_feature,axis=1)
feve_df['name_'] =[e[1] for e in feve_df.index]
feve_df = feve_df.groupby(['name_']).mean()
feve_df = feve_df.drop('RS02',axis=0)

feve_plot = plt.subplots(figsize=(6,4))
dev_metric = 'dev_f'
for feature_name in feature_names:
    col = f'{dev_metric}_{feature_name}'
    feve_plot[1].bar(col,feve_df[col].mean())
    feve_plot[1].scatter([col]*len(feve_df),feve_df[col],c='k',alpha=0.5)
[feve_plot[1].plot(row[[f'{dev_metric}_{f}' for f in feature_names]],c='grey',alpha=0.5)
 for _, row in feve_df.iterrows()]
feve_plot[1].set_xticklabels(feature_names)
feve_plot[1].set_ylabel('Deviance explained')
feve_plot[0].show()
feve_plot[0].savefig('dev_f_local_rate_vs_n_since.pdf')

