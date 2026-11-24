import argparse
from pathlib import Path
import warnings

import joblib
import pandas as pd
from tqdm import tqdm
import yaml

from analysis_scripts.aggregate_ephys_funcs import load_aggregate_td_df
from analysis_scripts.behviour_analysis_funcs import (
    get_all_cond_filts,
    get_main_sess_patterns,
    get_perturbation_sessions,
    group_td_df_across_sessions,
    parse_perturbation_log,
)
from analysis_scripts.io_utils import load_pupil_data, posix_from_win
from analysis_scripts.pupil_analysis_funcs import (
    add_name_to_response_dfs,
    group_pupil_across_sessions,
    init_pupil_td_obj,
    init_sess_pupil_obj,
    process_pupil_td_data,
)
from xdetectioncore.io_utils import save_response_dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save A/X by_cond data for arousal_control_opto.")
    parser.add_argument("config_file", default="config.yaml", nargs="?")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_file, "r"))
    ceph_dir = config["ceph_dir_windows"]
    home_dir = Path(config["home_dir_windows"])

    session_topology_paths = [
        ceph_dir / posix_from_win(r"X:\Xdetection_data\session_topologies\session_topology_opto_2503.csv"),
    ]
    session_topology_dfs = [pd.read_csv(p) for p in session_topology_paths]
    session_topology_df = pd.concat(session_topology_dfs, names=["cohort"]).reset_index()
    all_sess_info = session_topology_df.query('sess_order=="main" & Stage==4')

    conds2analyze = ["normal", "deviant_C"]
    good_trial_filt = "& n_since_last_Trial_Outcome <=5 & Trial_Outcome == 1 "
    cond_filters = get_all_cond_filts()

    td_dfs = load_aggregate_td_df(all_sess_info, home_dir)
    td_dfs = [td_dfs.query(cond_filters[cond]) for cond in conds2analyze]
    unique_sess_og = pd.concat(td_dfs, axis=0).index.get_level_values("sess").unique()
    unique_sess = [sess[:-1] for sess in unique_sess_og]

    drug_log_xlsx_path = ceph_dir / posix_from_win(r"X:\Dammy\Xdetection_mouse_hf_test\metadata\animal_log.xlsx")
    perturb_log = parse_perturbation_log(drug_log_xlsx_path)
    opto_sess_dict = get_perturbation_sessions(perturb_log["opto_log"], "opto_state")

    pupil_pkl_paths = [Path(r"D:\pupil\opto_2503_stage3_up_90Hz_hpass00_lpass0.joblib")]
    all_pupil_data = [load_pupil_data(pkl_path, sess_list=unique_sess) for pkl_path in pupil_pkl_paths]
    pupil_data_dict = {}
    for pupil_data in all_pupil_data:
        pupil_data_dict.update(pupil_data)

    sessions = {}
    for _s, sessname in tqdm(zip(unique_sess_og, unique_sess), desc="processing td"):
        sffx = ord(_s[-1]) - ord("a")
        init_pupil_td_obj(sessions, sessname, ceph_dir, all_sess_info, "", home_dir, sffx=sffx)
        if sessions[sessname].td_df.empty:
            sessions.pop(sessname)
            continue
        process_pupil_td_data(sessions, sessname, {})

    for _s, sessname in tqdm(zip(unique_sess_og, unique_sess), desc="processing pupil"):
        if sessname not in sessions:
            continue
        sffx = ord(_s[-1]) - ord("a")
        try:
            init_sess_pupil_obj(sessions, sessname, ceph_dir, all_sess_info, pupil_data_dict, sffx=sffx, force_sync=False)
        except KeyError:
            sessions.pop(sessname)
        except Exception as e:
            print(e)

    all_td_df = group_td_df_across_sessions(sessions, list(sessions.keys()))

    conds_by_epx_type = {
        "rate": ["rare", "frequent"],
        "oddball": ["normal", "deviant_C"],
        "abstraction": ["dev_ABBA1", "dev_ABCD1"],
    }
    exp_type_sess = {}
    for exp_type, conds in conds_by_epx_type.items():
        exp_td_sess = [all_td_df.query(cond_filters[cond]).index.get_level_values("sess").unique() for cond in conds]
        exp_type_sess[exp_type] = [
            sess for sess in exp_td_sess[0] if all(sess in _sess_list for _sess_list in exp_td_sess[1:])
        ]

    for sessname in sessions:
        main_patterns = get_main_sess_patterns(td_df=sessions[sessname].td_df)
        stage_val = sessions[sessname].td_df["Stage"].iloc[0]
        if stage_val in [3, 5]:
            normal_patterns = main_patterns
            if len(normal_patterns) > 1 and stage_val == 3:
                warnings.warn(f"{sessname} has more than one normal pattern for stage 3")
        elif stage_val == 4:
            normal_patterns = get_main_sess_patterns(td_df=sessions[sessname].td_df.query(cond_filters["normal_exp"]))
        else:
            continue

        normal = normal_patterns[0]
        events = {
            e: {"idx": idx, "filt": filt}
            for e, idx, filt in zip(["X_no_corr", "A", "X"], [3, normal[0], 3], ["", "pip_counter==1", ""])
        }
        for e, e_dict in events.items():
            sessions[sessname].get_pupil_to_event(
                e_dict["idx"],
                e,
                [-2, 0] if "no_corr" in e else [-1, 3],
                align_kwargs=dict(
                    sound_df_query=e_dict["filt"],
                    baseline_dur=0,
                    size_col="pupilsense_raddi_a_zscored",
                ),
                alignmethod="w_soundcard",
            )

        start_ids = list({patt[0] for patt in main_patterns})
        if len(start_ids) > 1:
            sessions[sessname].get_pupil_to_event(
                start_ids[1],
                "A_dev",
                [-1, 3],
                align_kwargs=dict(
                    sound_df_query="pip_counter==1",
                    baseline_dur=0,
                    size_col="pupilsense_raddi_a_zscored",
                ),
                alignmethod="w_soundcard",
            )

    A_by_cond = {
        cond: group_pupil_across_sessions(
            sessions,
            exp_type_sess["rate"]
            if cond in ["rare", "frequent"]
            else exp_type_sess["oddball"]
            if cond in ["normal", "deviant_C"]
            else exp_type_sess["abstraction"],
            "A" if cond not in ["dev_ABBA1", "dev_ABCD1"] else "A_dev",
            cond,
            td_df_query=cond_filters[cond] + good_trial_filt,
        ).droplevel("time")
        for cond in conds2analyze
    }
    X_by_cond = {
        cond: group_pupil_across_sessions(
            sessions,
            exp_type_sess["rate"]
            if cond in ["rare", "frequent"]
            else exp_type_sess["oddball"]
            if cond in ["normal", "deviant_C"]
            else exp_type_sess["abstraction"],
            "X",
            cond,
            td_df_query=cond_filters[cond] + good_trial_filt,
        ).droplevel("time")
        for cond in conds2analyze
    }
    X_pre_by_cond = {
        cond: group_pupil_across_sessions(
            sessions,
            exp_type_sess["rate"]
            if cond in ["rare", "frequent"]
            else exp_type_sess["oddball"]
            if cond in ["normal", "deviant_C"]
            else exp_type_sess["abstraction"],
            "X_no_corr",
            cond,
            td_df_query=cond_filters[cond] + good_trial_filt,
        )
        for cond in conds2analyze
    }

    for stim_by_cond in [A_by_cond, X_by_cond, X_pre_by_cond]:
        add_name_to_response_dfs(stim_by_cond)

    conds_by_expected = {
        "expected": ["normal"],
        "unexpected": ["deviant_C"],
    }
    for stim_by_cond in [A_by_cond, X_by_cond, X_pre_by_cond]:
        for k, v in conds_by_expected.items():
            stim_by_cond[k] = pd.concat([stim_by_cond[c] for c in v])

    joblib.dump(A_by_cond, "fig2_opto_A_pre_by_cond.pkl")
    joblib.dump(X_by_cond, "fig2_opto_X_by_cond.pkl")
    joblib.dump(X_pre_by_cond, "fig2_opto_X_pre_by_cond.pkl")
    save_response_dfs(A_by_cond, "fig2_opto_A_pre_by_cond.h5")
    save_response_dfs(X_pre_by_cond, "fig2_opto_X_pre_by_cond.h5")