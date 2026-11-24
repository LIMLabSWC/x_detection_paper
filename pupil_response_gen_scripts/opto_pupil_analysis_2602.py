import argparse
from copy import copy
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from analysis_scripts.pupil_analysis_funcs import (
    add_name_to_response_dfs,
    group_pupil_across_sessions,
    init_pupil_td_obj,
    init_sess_pupil_obj,
    load_pupil_sess_pkl,
    process_pupil_obj,
    process_pupil_td_data,
)
from xdetectioncore.behaviour import (
    get_all_cond_filts,
    get_main_sess_patterns,
    get_perturbation_sessions,
    group_td_df_across_sessions,
    parse_perturbation_log,
)
from xdetectioncore.io_utils import extract_date, load_pupil_sess_lazy, save_response_dfs
from xdetectioncore.paths import posix_from_win
from xdetectioncore.session import Session


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save A/X by_cond data for opto_2602.")
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
        ceph_dir / posix_from_win(r"X:\Xdetection_data\session_topologies\session_topology_opto_2602.csv"),
    ]
    pupil_pkl_paths = [
        Path(p)
        for p in [
            r"X:\Dammy\mouse_pupillometry\pickles\opto_2602_90Hz_hpass005_lpass0",
        ]
    ]

    session_topology_dfs = [pd.read_csv(p) for p in session_topology_paths]
    session_topology_df = pd.concat(session_topology_dfs, names=["cohort"]).reset_index()
    session_topology_df = session_topology_df.query('name != "RS13"')
    all_sess_info = session_topology_df[session_topology_df["status"] == "Complete"]

    sess_pkl_path = Path(r"D:\opto_2602_w_pupilsense_analysis.joblib")
    sessions: dict = load_pupil_sess_pkl(sess_pkl_path)
    loaded_sess_dict_sig = copy(tuple(sorted(sessions.items())))

    pupil_lazy_loader = load_pupil_sess_lazy(pupil_pkl_paths[0])
    available_sessions = pupil_lazy_loader.keys()

    A_by_cond_path = Path(r"D:\opto_2602_A_by_cond.joblib")
    X_by_cond_path = Path(r"D:\opto_2602_X_by_cond.joblib")
    redo_alignment = [True, True]

    if any(redo_alignment) or not A_by_cond_path.is_file():
        for sessname in tqdm(available_sessions, desc="processing sessions"):
            if int(extract_date(sessname)) < 260401 or "RS13" in sessname:
                continue

            if sessname not in sessions:
                sessions[sessname] = Session(sessname, ceph_dir, None)

            if any([sessions[sessname].td_df is not None, not sessions[sessname].td_df.empty]) and not args.ow:
                if sessions[sessname].pupil_obj is not None:
                    continue

            try:
                if any([sessions[sessname].td_df is not None, not sessions[sessname].td_df.empty]) or args.ow:
                    init_pupil_td_obj(sessions, sessname, ceph_dir, all_sess_info, td_path_pattern, home_dir)
                    process_pupil_td_data(sessions, sessname, {})
            except AttributeError as e:
                print(f"Error TrialData for {sessname}: {e}")
                sessions.pop(sessname, None)
                continue

            if sessions[sessname].td_df.empty or len(sessions[sessname].td_df) < 50:
                print(f"Insufficient data for {sessname}")
                sessions.pop(sessname, None)
                continue

            try:
                init_sess_pupil_obj(sessions, sessname, ceph_dir, all_sess_info, pupil_lazy_loader)
                process_pupil_obj(
                    sessions,
                    sessname,
                    alignmethod="w_soundcard",
                    align_kwargs={"size_col": "pupilsense_raddi_a_zscored"},
                )
            except Exception as e:
                print(f"Error processing {sessname}: {e}")
                sessions.pop(sessname, None)

    window = [-1, 3]
    drug_log_xlsx_path = r"X:\Dammy\Xdetection_mouse_hf_test\metadata\animal_log.xlsx"
    perturb_log = parse_perturbation_log(drug_log_xlsx_path)
    opto_sess_dict = get_perturbation_sessions(perturb_log["opto_log"], "opto_state")
    if "session_long_light_on" in opto_sess_dict:
        opto_sess_dict["light_on"] = list(
            set(opto_sess_dict["light_on"] + opto_sess_dict["session_long_light_on"])
        )
        del opto_sess_dict["session_long_light_on"]

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
    X_by_cond["normal"] = group_pupil_across_sessions(sessions, list(sessions.keys()), "X", "normal", cond_filters)
    for cond in dev_cond_names[1:]:
        A_by_cond[cond] = group_pupil_across_sessions(sessions, a_dev_sess, "A_dev", cond_name=cond, cond_filters=cond_filters)
        X_by_cond[cond] = group_pupil_across_sessions(sessions, a_dev_sess, "X", cond_name=cond, cond_filters=cond_filters)

    add_name_to_response_dfs(A_by_cond)
    add_name_to_response_dfs(X_by_cond)

    for resp_path, resps_dict in zip([A_by_cond_path, X_by_cond_path], [A_by_cond, X_by_cond]):
        resp_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(resps_dict, resp_path)

    save_response_dfs(A_by_cond, A_by_cond_path.with_suffix(".h5"))
    save_response_dfs(X_by_cond, X_by_cond_path.with_suffix(".h5"))