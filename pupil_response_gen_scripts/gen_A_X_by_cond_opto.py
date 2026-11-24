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
    get_perturbation_sessions,
    group_td_df_across_sessions,
    parse_perturbation_log,
)
from analysis_scripts.io_utils import extract_date, load_pupil_data, posix_from_win
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
    parser = argparse.ArgumentParser(description="Generate and save A/X by_cond data for opto_pupil_analysis.")
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
        ceph_dir / posix_from_win(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_opto_2503.csv"),
        ceph_dir / posix_from_win(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2504.csv"),
    ]
    pupil_pkl_paths = [
        Path(p)
        for p in [
            r"X:\Dammy\mouse_pupillometry\pickles\opto_2503_stage3_up_90Hz_hpass00_lpass0.joblib",
            r"X:\Dammy\mouse_pupillometry\pickles\ephys_2504_stage3_up_90Hz_hpass00_lpass0.joblib",
        ]
    ]

    session_topology_dfs = [pd.read_csv(p) for p in session_topology_paths]
    session_topology_df = pd.concat(session_topology_dfs, names=["cohort"]).reset_index()
    all_sess_info = session_topology_df

    sess_pkl_path = Path(r"D:\all_opto_w_pupilsense_analysis.joblib")
    sessions: dict = load_pupil_sess_pkl(sess_pkl_path)
    loaded_sess_dict_sig = copy(tuple(sorted(sessions.items())))

    drug_log_xlsx_path = r"X:\Dammy\Xdetection_mouse_hf_test\metadata\animal_log.xlsx"
    perturb_log = parse_perturbation_log(drug_log_xlsx_path)
    opto_sess_dict = get_perturbation_sessions(perturb_log["opto_log"], "opto_state")
    if "session_long_light_on" in opto_sess_dict:
        opto_sess_dict["light_on"] = list(
            set(opto_sess_dict["light_on"] + opto_sess_dict["session_long_light_on"])
        )
        del opto_sess_dict["session_long_light_on"]

    last_loaded_date = int(sorted([extract_date(sess) for sess in sessions.keys()])[-1]) if len(sessions) > 0 else 0

    if len(sessions) > 1e12:
        all_pupil_data = [load_pupil_data(pkl_path) for pkl_path in pupil_pkl_paths]
        pupil_data_dict = all_pupil_data[0]
        for data in all_pupil_data[1:]:
            pupil_data_dict.update(data)

        for sessname in tqdm(list(pupil_data_dict.keys()), desc="processing td"):
            if int(extract_date(sessname)) < last_loaded_date:
                continue
            if sessname not in sessions:
                sessions[sessname] = Session(sessname, ceph_dir)
            if sessions[sessname].td_df is not None and not args.ow:
                continue
            init_pupil_td_obj(sessions, sessname, ceph_dir, all_sess_info, td_path_pattern, home_dir)
            process_pupil_td_data(sessions, sessname, {})

        for sessname in tqdm(list(sessions.keys()), desc="processing pupil"):
            if sessions[sessname].pupil_obj is not None and not args.ow:
                continue
            try:
                init_sess_pupil_obj(sessions, sessname, ceph_dir, all_sess_info, pupil_data_dict)
                process_pupil_obj(
                    sessions,
                    sessname,
                    alignmethod="w_soundcard",
                    align_kwargs={"size_col": "pupilsense_raddi_a_zscored"},
                )
            except Exception as e:
                print(f"error processing {sessname}: {e}")
                sessions.pop(sessname, None)

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
        for cond in ["hit_all", "miss_all"]
    }

    rare_freq_early_late_dict = {"early": "<=8", "middle": ">9 & <col> <=17", "late": ">19 & frequent_block_num==1"}
    add_early_late_to_by_cond(A_by_cond, ["frequent_prate"], rare_freq_early_late_dict, sessions, "A", cond_filters)

    rare_freq_condnames = ["frequent", "rare"]
    all_rare_freq_sess = [
        sess for sess in sessions if all(sess in A_by_cond[c].index.get_level_values("sess") for c in rare_freq_condnames)
    ]
    cond_filters["pattern"] = "Tone_Position==0 & Stage==3 & N_TonesPlayed==4 & Session_Block==0"
    A_by_cond["pattern"] = group_pupil_across_sessions(sessions, all_rare_freq_sess, "A", "pattern", cond_filters)

    norm_dev_cond_names = ["normal", "deviant_C"]
    for cond in norm_dev_cond_names:
        cond_filters[cond] = cond_filters[cond] + " & Trial_Outcome==1"
        A_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), "A", cond, cond_filters)
        X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), "X", cond, cond_filters)
    add_name_to_response_dfs(A_by_cond)

    for cond in rare_freq_condnames:
        X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), "X", cond, cond_filters)

    cond_filters["frequent_late"] = "frequent_prate_cumsum >= 25  & frequent_block_num == 2"
    rare_freq_early_late_conds = [f"frequent_prate_{ti}_block_1" for ti in np.arange(0, 25, 5)][:5]
    for cond in rare_freq_early_late_conds:
        A_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), "A", cond, cond_filters)
        X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), "X", cond, cond_filters)

    output_dir = ceph_dir / posix_from_win(r"X:\Xdetection_data\data_for_paper")
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(A_by_cond, output_dir / "fig2_opto_A_by_cond.pkl")
    joblib.dump(X_by_cond, output_dir / "fig2_opto_X_by_cond.pkl")
    all_td_df.to_hdf(output_dir / "fig2_opto_all_td_df.h5", key="all_td_df")
    with open(output_dir / "opto_sess_dict.json", "w") as f:
        json.dump(opto_sess_dict, f)