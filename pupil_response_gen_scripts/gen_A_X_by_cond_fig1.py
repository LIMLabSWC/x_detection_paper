import argparse
from copy import copy
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from analysis_scripts.behviour_analysis_funcs import (
    get_all_cond_filts,
    get_main_sess_patterns,
    get_perturbation_sessions,
    group_td_df_across_sessions,
    parse_perturbation_log,
)
from analysis_scripts.io_utils import load_pupil_data, posix_from_win
from analysis_scripts.pupil_analysis_funcs import (
    add_early_late_to_by_cond,
    add_name_to_response_dfs,
    group_pupil_across_sessions,
    init_pupil_td_obj,
    init_sess_pupil_obj,
    load_pupil_sess_pkl,
    process_pupil_obj,
    process_pupil_td_data,
)
from analysis_scripts.sess_dataclasses import Session


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save A/X by_cond data for ch_2 analysis.")
    parser.add_argument("config_file", default="config.yaml", nargs="?")
    parser.add_argument("--ow", default=0, type=int)
    args = parser.parse_args()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    sys_os = platform.system().lower()
    ceph_dir = Path(config[f"ceph_dir_{sys_os}"])
    home_dir = Path(config[f"home_dir_{sys_os}"])
    td_path_pattern = "data/Hilde/<name>/TrialData"

    session_topology_paths = [
        ceph_dir / posix_from_win(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_2406.csv"),
        ceph_dir / posix_from_win(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2401.csv"),
    ]
    pupil_pkl_paths = [
        Path(r"D:\pupil\musc_2406_stage3_up_90Hz_hpass0_lpass0.joblib"),
        Path(r"D:\pupil\ephys_2401_stage3_up_90Hz_hpass00_lpass0.joblib"),
    ]

    session_topology_dfs = [pd.read_csv(p) for p in session_topology_paths]
    session_topology_df = pd.concat(session_topology_dfs, names=["cohort"]).reset_index()
    all_sess_info = session_topology_df

    sess_pkl_path = (
        Path(r"D:\all_w_stage4.joblib")
        if sys_os == "windows"
        else ceph_dir / posix_from_win(r"X:\Dammy\pupil_data\all_w_stage4.joblib")
    )
    sessions: dict = load_pupil_sess_pkl(sess_pkl_path)
    loaded_sess_dict_sig = copy(tuple(sorted(sessions.items())))

    window = [-1, 3]
    drug_log_xlsx_path = ceph_dir / posix_from_win(r"X:\Dammy\Xdetection_mouse_hf_test\metadata\animal_log.xlsx")
    perturb_log = parse_perturbation_log(drug_log_xlsx_path)
    drug_sess_dict = get_perturbation_sessions(perturb_log["infusion_log"], "drug")

    if len(sessions) == 0:
        all_pupil_data = [load_pupil_data(pkl_path) for pkl_path in pupil_pkl_paths]
        pupil_data_dict = {**all_pupil_data[0], **all_pupil_data[1]}

        for sessname in tqdm(list(pupil_data_dict.keys()), desc="processing td"):
            if sessname not in sessions:
                sessions[sessname] = Session(sessname, ceph_dir)
            if sessions[sessname].td_df is not None and not args.ow:
                continue
            init_pupil_td_obj(sessions, sessname, ceph_dir, all_sess_info, td_path_pattern, home_dir)
            process_pupil_td_data(sessions, sessname, {})

        for sessname in tqdm(list(sessions.keys()), desc="processing pupil"):
            if sessions[sessname].pupil_obj is not None and not args.ow:
                continue
            init_sess_pupil_obj(sessions, sessname, ceph_dir, all_sess_info, pupil_data_dict)
            process_pupil_obj(
                sessions,
                sessname,
                alignmethod="w_soundcard",
                align_kwargs={"size_col": "pupilsense_raddi_a_zscored"},
            )

        joblib.dump(sessions, sess_pkl_path)

    cond_filters = get_all_cond_filts()
    all_td_df = group_td_df_across_sessions(sessions, list(sessions.keys()))
    all_td_df["PatternPresentation_Rate"] = all_td_df["PatternPresentation_Rate"].round(1)

    conds2analyze = ["rare_prate", "frequent_prate", "rare", "frequent"]
    A_by_cond = {
        cond: group_pupil_across_sessions(sessions, list(sessions.keys()), "A", cond, cond_filters)
        for cond in conds2analyze
    }
    X_by_cond = {
        cond: group_pupil_across_sessions(sessions, list(sessions.keys()), "X", cond, cond_filters)
        for cond in ["hit_all", "miss_all"] + conds2analyze
    }

    rare_freq_early_late_dict = {"early": "<=8", "middle": ">9 & <col> <=17", "late": ">19 & frequent_block_num==1"}
    add_early_late_to_by_cond(A_by_cond, ["frequent_prate"], rare_freq_early_late_dict, sessions, "A", cond_filters)

    for cond in ["AB", "ABC", "ABCD"]:
        X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), "X", cond, cond_filters)

    rare_freq_condnames = ["frequent", "rare"]
    for cond in rare_freq_condnames:
        if cond not in X_by_cond:
            X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), "X", cond, cond_filters)

    all_rare_freq_sess = [
        sess for sess in sessions if all(sess in A_by_cond[c].index.get_level_values("sess") for c in rare_freq_condnames)
    ]
    drug_rare_freq_sesslist = [
        sess
        for sess in all_rare_freq_sess
        if sess not in drug_sess_dict.get("muscimol", []) and sess not in drug_sess_dict.get("saline", [])
    ]

    norm_dev_cond_names = ["normal", "deviant_C"]
    for cond in norm_dev_cond_names:
        A_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), "A", cond, cond_filters)
        X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), "X", cond, cond_filters)

    norm_dev_sess = [
        sess for sess in sessions if all(sess in A_by_cond[c].index.get_level_values("sess") for c in norm_dev_cond_names)
    ]
    norm_dev_sesslist = [
        sess
        for sess in norm_dev_sess
        if sess not in drug_sess_dict.get("muscimol", []) and sess not in drug_sess_dict.get("saline", [])
    ]

    dev_cond_names = ["normal", "dev_ABCD1", "dev_ABBA1"]
    for sessname in sessions:
        main_patterns = get_main_sess_patterns(td_df=sessions[sessname].td_df)
        start_ids = list({patt[0] for patt in main_patterns})
        if len(start_ids) <= 1:
            continue
        events = {"A_dev": {"idx": start_ids[1], "filt": "pip_counter==1"}}
        for e, e_dict in events.items():
            sessions[sessname].get_pupil_to_event(
                e_dict["idx"],
                e,
                window,
                align_kwargs=dict(
                    sound_df_query=e_dict["filt"],
                    baseline_dur=1,
                    size_col="pupilsense_raddi_a_zscored",
                ),
                alignmethod="w_soundcard",
            )

    a_dev_sess = [sess for sess in sessions if "A_dev" in sessions[sess].pupil_obj.aligned_pupil]
    A_by_cond["normal"] = group_pupil_across_sessions(sessions, list(sessions.keys()), "A", "normal", cond_filters)
    for cond in dev_cond_names[1:]:
        A_by_cond[cond] = group_pupil_across_sessions(sessions, a_dev_sess, "A_dev", cond_name=cond, cond_filters=cond_filters)

    add_name_to_response_dfs(A_by_cond)
    dev_sess = [
        sess for sess in sessions if all(sess in A_by_cond[c].index.get_level_values("sess") for c in dev_cond_names)
    ]
    dev_sesslist = [
        sess
        for sess in dev_sess
        if sess not in drug_sess_dict.get("muscimol", []) and sess not in drug_sess_dict.get("saline", [])
    ]

    cond_filters["frequent_late"] = "frequent_prate_cumsum >= 25  & frequent_block_num == 2"
    rare_freq_early_late_conds = [f"frequent_prate_{ti}" for ti in np.arange(0, 25, 5)][:3][::2]
    for cond in rare_freq_early_late_conds:
        A_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), "A", cond, cond_filters)
        X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), "X", cond, cond_filters)

    output_dir = ceph_dir / posix_from_win(r"X:\Xdetection_data\data_for_paper")
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(A_by_cond, output_dir / "fig1_A_by_cond.pkl")
    joblib.dump(X_by_cond, output_dir / "fig1_X_by_cond.pkl")
    all_td_df.to_hdf(output_dir / "all_td_df.h5", key="all_td_df")

    sesslist_by_cond = {
        condname: con_sess_list
        for condname, con_sess_list in zip(
            ["event_rate", "norm_dev", "generalisation"],
            [drug_rare_freq_sesslist, norm_dev_sesslist, dev_sesslist],
        )
    }
    with open(output_dir / "sesslist_by_cond.json", "w") as f:
        json.dump(sesslist_by_cond, f)