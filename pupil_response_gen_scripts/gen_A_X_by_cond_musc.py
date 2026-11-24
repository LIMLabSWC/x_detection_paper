import argparse
from copy import copy
from datetime import datetime
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from behviour_analysis_funcs import (
    get_all_cond_filts,
    get_perturbation_sessions,
    group_td_df_across_sessions,
    parse_perturbation_log,
)
from io_utils import load_pupil_data, posix_from_win
from pupil_analysis_funcs import (
    add_early_late_to_by_cond,
    group_pupil_across_sessions,
    init_pupil_td_obj,
    init_sess_pupil_obj,
    load_pupil_sess_pkl,
    process_pupil_obj,
    process_pupil_td_data,
)
from sess_dataclasses import Session


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save A/X by_cond data for musc_pupil_analysis.")
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
        ceph_dir / posix_from_win(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_sept23.csv"),
        ceph_dir / posix_from_win(r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_musc_2406.csv"),
    ]
    pupil_pkl_paths = [
        ceph_dir / posix_from_win(p)
        for p in [
            r"X:\Dammy\mouse_pupillometry\pickles\musc_sept23_stage3_up_90Hz_hpass00_lpass0.joblib",
            r"X:\Dammy\mouse_pupillometry\pickles\musc_2406_stage3_up_90Hz_hpass0_lpass0.joblib",
        ]
    ]

    sess_pkl_path = Path(r"D:\musc_analysis.joblib")
    sessions: dict = load_pupil_sess_pkl(sess_pkl_path)
    loaded_sess_dict_sig = copy(tuple(sorted(sessions.items())))

    session_topology_dfs = [pd.read_csv(p) for p in session_topology_paths]
    session_topology_df = pd.concat(session_topology_dfs, names=["cohort"]).reset_index()
    all_sess_info = session_topology_df

    drug_log_xlsx_path = r"X:\Dammy\Xdetection_mouse_hf_test\metadata\animal_log.xlsx"
    perturb_log = parse_perturbation_log(drug_log_xlsx_path)
    drug_sess_dict = get_perturbation_sessions(perturb_log["infusion_log"], "drug")

    exclude_dates = ["240710", "240712", "240716", "240717", "240726"]
    none_sess = drug_sess_dict["none"]
    musc_sess = drug_sess_dict["muscimol"]
    saline_sess = drug_sess_dict["saline"]
    none_dates = list({sess.split("_")[1] for sess in none_sess})
    musc_dates = list({sess.split("_")[1] for sess in musc_sess})
    saline_dates = list({sess.split("_")[1] for sess in saline_sess})
    none_dates_dt, musc_dates_dt, saline_dates_dt = [
        [datetime.strptime(d, "%y%m%d") for d in dates]
        for dates in [none_dates, musc_dates, saline_dates]
    ]
    _, saline_dates = [
        [d for d, d_dt in zip(date_set, dt_set) if all(d_dt - np.array(musc_dates_dt) != np.timedelta64(1, "D"))]
        for date_set, dt_set in zip([none_dates, saline_dates], [none_dates_dt, saline_dates_dt])
    ]
    drug_sess_dict["muscimol"] = [
        sess for sess in drug_sess_dict["muscimol"] if sess.split("_")[1] not in exclude_dates
    ]
    drug_sess_dict["saline"] = [
        sess
        for sess in drug_sess_dict["saline"]
        if sess.split("_")[1] in saline_dates and sess.split("_")[1] not in exclude_dates
    ]

    if len(sessions) == 0:
        all_pupil_data = [load_pupil_data(pkl_path) for pkl_path in pupil_pkl_paths]
        pupil_data_dict = {**all_pupil_data[0], **all_pupil_data[1]}

        for sessname in tqdm(list(pupil_data_dict.keys()), desc="processing td"):
            if sessname not in sessions:
                sessions[sessname] = Session(sessname, ceph_dir)
            if sessions[sessname].td_df is not None and not args.ow:
                continue
            init_pupil_td_obj(sessions, sessname, ceph_dir, all_sess_info, td_path_pattern, home_dir)
            process_pupil_td_data(sessions, sessname, drug_sess_dict)

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
        for cond in ["hit_all", "miss_all"]
    }

    rare_freq_condnames = ["frequent", "rare"]
    for cond in rare_freq_condnames:
        X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), "X", cond, cond_filters)

    cond_filters["frequent_late"] = "frequent_prate_cumsum >= 25  & frequent_block_num == 2"
    rare_freq_early_late_conds = [f"frequent_prate_{ti}_block_1" for ti in np.arange(0, 25, 5)][:5]
    for cond in rare_freq_early_late_conds:
        A_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), "A", cond, cond_filters)
        X_by_cond[cond] = group_pupil_across_sessions(sessions, list(sessions.keys()), "X", cond, cond_filters)

    output_dir = ceph_dir / posix_from_win(r"X:\Xdetection_data\data_for_paper")
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(A_by_cond, output_dir / "fig2_musc_A_by_cond.pkl")
    joblib.dump(X_by_cond, output_dir / "fig2_musc_X_by_cond.pkl")
    all_td_df.to_hdf(output_dir / "fig2_musc_all_td_df.h5", key="all_td_df")
    with open(output_dir / "musc_sess_dict.json", "w") as f:
        json.dump(drug_sess_dict, f)