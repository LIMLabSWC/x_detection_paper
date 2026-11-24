import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
import yaml

from behviour_analysis_funcs import (
    get_all_cond_filts,
    get_main_sess_patterns,
    get_main_sess_td_df,
)

from ephys_analysis_funcs import gen_metadata, get_patts_by_rule, get_pip_info
from io_utils import load_sess_pkl, posix_from_win
from sess_dataclasses import Session


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # 1. CLI Arguments & Config Setup
    # ---------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Build and save per-session ephys pickles to ephys_concat_pkls."
    )
    parser.add_argument("config_file", help="Path to YAML configuration file.")
    parser.add_argument(
        "sess_date", help="Target subject and date formatted as '<name>_<yymmdd>'."
    )
    parser.add_argument("--sorter_dirname", default="from_concat", required=False)
    parser.add_argument("--sess_top_tag", default="")
    parser.add_argument("--sess_top_filts", default="")
    parser.add_argument("--synth_data", default=0, type=int)
    parser.add_argument("--rel_sorting_path", default="")
    parser.add_argument(
        "--ow",
        default=1,
        type=int,
        help="If 1, overwrite existing pickles and re-initialize sound events.",
    )

    args = parser.parse_args()
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    # Resolve OS-specific root paths and output directory for session pickles
    sys_os = platform.system().lower()
    ceph_dir = Path(config[f"ceph_dir_{sys_os}"])
    home_dir = Path(config[f"home_dir_{sys_os}"])
    pkl_dir = ceph_dir / "Dammy" / "ephys_concat_pkls"
    pkl_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 2. Load Session Topology & Filter Target Subject/Date
    # ---------------------------------------------------------------------
    sess_topology_path = ceph_dir / posix_from_win(
        rf"X:\Dammy\Xdetection_mouse_hf_test\session_topology_{args.sess_top_tag}.csv"
    )
    # Ensure behavioral/metadata columns exist before filtering
    gen_metadata(sess_topology_path, ceph_dir, col_name="beh_bin", harp_bin_dir="")
    session_topology = pd.read_csv(sess_topology_path)

    name, date_str_raw = args.sess_date.split("_")
    date = int(date_str_raw)
    all_sess_info = session_topology.query("name==@name & date==@date").reset_index(
        drop=True
    )
    if args.sess_top_filts:
        all_sess_info = all_sess_info.query(args.sess_top_filts)

    # Determine sorting subdirectory hierarchy
    if args.rel_sorting_path:
        dir1_name, dir2_name = Path(args.rel_sorting_path).parts
    else:
        dir1_name, dir2_name = "sorting_no_si_drift", "kilosort2_5_ks_drift"

    # ---------------------------------------------------------------------
    # 3. Resolve Main Session TrialData & Baseline Sequence Patterns
    # ---------------------------------------------------------------------
    psth_window = [-2, 3]  # Seconds relative to event onset: [pre, post]
    main_sess = session_topology.query('sess_order=="main" & name==@name & date==@date')
    if "tdata_file" in main_sess.columns:
        main_sess_td_name = posix_from_win(
            main_sess["tdata_file"].iloc[0], ceph_linux_dir=config["home_dir_linux"]
        )
    else:
        main_sess_td_name = (
            Path(main_sess["sound_bin"].iloc[0].replace("_SoundData", "_TrialData"))
            .with_suffix(".csv")
            .name
        )

    # Fallback to dummy pattern if no TrialData file is found yet
    try:
        main_patterns = get_main_sess_patterns(
            name, date, main_sess_td_name=main_sess_td_name, home_dir=home_dir
        )
    except FileNotFoundError:
        main_patterns = [[0, 0, 0, 0]]

    main_pattern = main_patterns[0]
    cond_filters = get_all_cond_filts()

    spike_cluster_path = "spike_clusters.npy"
    spike_times_path = "spike_times.npy"

    # ---------------------------------------------------------------------
    # 4. Process Each Recording/Session & Dump Pickle to `pkl_dir`
    # ---------------------------------------------------------------------
    for _, sess_info in all_sess_info.iterrows():
        spike_dir = (
            ceph_dir
            / posix_from_win(sess_info["ephys_dir"])
            / args.rel_sorting_path
            / args.sorter_dirname
        )
        # Locate OpenEphys recording directory containing 'continuous' and metadata.json
        recording_dir = next(
            (
                ceph_dir
                / posix_from_win(sess_info["ephys_dir"], config["ceph_dir_linux"])
            ).rglob("continuous")
        ).parent
        print(f"analysing {recording_dir.name}")

        with open(recording_dir / "metadata.json", "r") as jsonfile:
            recording_meta = json.load(jsonfile)
        start_time = recording_meta["trigger_time"]

        # Derive canonical session name used for pickle filenames
        sessname = Path(sess_info["sound_bin"]).stem.replace("_SoundData", "")
        sess_pkl_path = pkl_dir / f"{sessname}.pkl"

        # Load cached Session object if present and overwrite (--ow) is off
        if sess_pkl_path.is_file() and not args.ow:
            sess_obj = load_sess_pkl(sess_pkl_path)
        else:
            sess_obj = Session(sessname, ceph_dir)

        sound_bin_path = Path(sess_info["sound_bin"])
        beh_bin_path = Path(sess_info["beh_bin"])

        # Attach trial behavioral table and skip early stage-2 main sessions
        sess_obj.load_trial_data(
            get_main_sess_td_df(name, date, main_sess_td_name, home_dir)[1]
        )
        if sess_obj.td_df["Stage"].iloc[0] <= 2 and sess_info["sess_order"] == "main":
            print("stage 2, nothing to analyse")
            continue

        # Sort sequence patterns and determine stage-specific normal patterns
        main_patterns = get_main_sess_patterns(td_df=sess_obj.td_df)
        main_patterns = sorted(
            main_patterns, key=lambda x: (x[0], np.any(np.diff(x) <= 0))
        )
        patts_by_rule = get_patts_by_rule(main_patterns)
        n_patts_per_rule = max(len(e) for e in patts_by_rule.values())

        stage_val = sess_obj.td_df["Stage"].iloc[0]
        if stage_val == 3:
            normal_patterns = main_patterns
            if len(normal_patterns) > 1:
                warnings.warn(
                    f"{sessname} has more than one normal pattern for stage 3"
                )
        elif stage_val == 4:
            normal_patterns = get_main_sess_patterns(
                td_df=sess_obj.td_df.query(cond_filters["normal_exp"])
            )
        elif stage_val == 5:
            normal_patterns = [e for i, e in enumerate(main_patterns) if i % 2 == 0]
        else:
            normal_patterns = main_patterns

        # Initialize spike sorting data aligned to behavioral hardware write events
        beh_write_data_path = beh_bin_path.with_stem(
            f"{beh_bin_path.stem}_write_data"
        ).with_suffix(".csv")
        beh_write_data_path = ceph_dir / posix_from_win(str(beh_write_data_path))

        sess_obj.init_spike_obj(
            spike_times_path,
            spike_cluster_path,
            start_time,
            parent_dir=spike_dir,
            beh_write_data_path=beh_write_data_path,
            rec_dir=recording_dir,
        )

        # Populate sound_event_dict from sound hardware write indices if needed
        if not sess_obj.sound_event_dict or args.ow:
            abs_writes_path = str(
                sound_bin_path.with_stem(f"{sound_bin_path.stem}_write_indices")
            )
            sess_obj.init_sound_event_dict(
                ceph_dir / posix_from_win(abs_writes_path).with_suffix(".csv"),
                patterns=main_patterns,
                normal_patterns=normal_patterns,
            )

        # Compute event-free baseline z-score statistics across units
        sess_obj.get_event_free_zscore()

        # Optionally replace unit mean/sd with unit-variance/zero-mean for synthetic tests
        synth_data_flag = args.synth_data if args.synth_data else None
        n_units = len(sess_obj.spike_obj.units)
        if synth_data_flag:
            sess_obj.spike_obj.unit_means = (np.zeros(n_units), np.ones(n_units))

        # Build pip metadata descriptors and store on the Session object
        pip_idxs = {
            event_lbl: sess_obj.sound_event_dict[event_lbl].idx
            for event_lbl in sess_obj.sound_event_dict
            if any(char in event_lbl for char in "ABCD")
        }
        pip_desc, _, _ = get_pip_info(
            sess_obj.sound_event_dict,
            main_patterns
            if sess_info["sess_order"] == "main"
            else list(pip_idxs.values()),
            n_patts_per_rule,
        )
        sess_obj.pip_desc = pip_desc

        # Compute PSTHs for all sound events and serialize the Session to `pkl_dir`
        sess_obj.get_sound_psth(
            psth_window=psth_window,
            zscore_flag=False,
            baseline_dur=0,
            redo_psth=False,
            use_iti_zscore=False,
            synth_data=synth_data_flag,
        )
        sess_obj.pickle_obj(pkl_dir)