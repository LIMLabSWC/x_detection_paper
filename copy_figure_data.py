from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


SOURCE_ROOT = Path(r"H:\ephys")
DESTINATION_ROOT = Path(r"H:\x_detection_paper") / "data"

# Paths that are not absolute are resolved relative to SOURCE_ROOT.
FILES = [
    # Figure 1
    (r"X:\Xdetection_data\data_for_paper\fig1_A_by_cond.h5",
     "pupil_responses/fig01_mouse_A_pupil_by_condition.h5"),
    (r"X:\Xdetection_data\data_for_paper\fig1_X_by_cond.h5",
     "pupil_responses/fig01_mouse_X_pupil_by_condition.h5"),
    (r"X:\Xdetection_data\data_for_paper\all_td_df.h5",
     "trial_tables/fig01_mouse_trials.h5"),
    ("all_td_df_4_beh.h5",
     "trial_tables/fig01_supp_behavior_trials.h5"),
    ("patt_non_patt_licks_df.h5",
     "behavior/fig01_lick_traces_pattern_vs_nonpattern.h5"),
    ("patt_non_patt2X_licks_df.h5",
     "behavior/fig01_lick_traces_pattern_vs_X.h5"),
    ("arousal_control_A_by_cond.h5",
     "pupil_responses/fig01_arousal_control_A_pupil_by_condition.h5"),
    ("arousal_control_X_by_cond.h5",
     "pupil_responses/fig01_arousal_control_X_pupil_by_condition.h5"),
    ("fig1_human_A_by_cond.h5",
     "human_responses/fig01_human_A_pupil_by_condition.h5"),
    ("fig1_human_X_by_cond.h5",
     "human_responses/fig01_human_X_pupil_by_condition.h5"),
    ("sesslist_by_cond.json",
     "session_metadata/fig01_sessions_by_analysis.json"),
    ("hazard_rate_td_df.h5",
     "trial_tables/supp_fig01_hazard_rate_trials.h5"),

    # Figure 2 and Figure 4 optogenetic data
    (r"X:\Xdetection_data\data_for_paper\fig2_musc_A_by_cond.h5",
     "pupil_responses/fig02_muscimol_A_pupil_by_condition.h5"),
    (r"X:\Xdetection_data\data_for_paper\fig2_musc_X_by_cond.h5",
     "pupil_responses/fig02_muscimol_X_pupil_by_condition.h5"),
    (r"X:\Xdetection_data\data_for_paper\fig2_musc_all_td_df.h5",
     "trial_tables/fig02_muscimol_trials.h5"),
    (r"X:\Xdetection_data\data_for_paper\musc_sess_dict.json",
     "session_metadata/fig02_muscimol_sessions_by_state.json"),
    (r"X:\Xdetection_data\data_for_paper\fig2_opto_A_by_cond.h5",
     "pupil_responses/fig02_opto_A_pupil_by_condition.h5"),
    (r"X:\Xdetection_data\data_for_paper\fig2_opto_X_by_cond.h5",
     "pupil_responses/fig02_opto_X_pupil_by_condition.h5"),
    (r"X:\Xdetection_data\data_for_paper\td_df_for_norm_dev.h5",
     "trial_tables/fig02_opto_normdev_trials.h5"),
    (r"X:\Xdetection_data\data_for_paper\fig2_opto_all_td_df.h5",
     "trial_tables/fig04_opto_trials.h5"),
    (r"X:\Xdetection_data\data_for_paper\opto_sess_dict.json",
     "session_metadata/fig02_fig04_opto_sessions_by_state.json"),
    (r"D:\paper_data\fig2\gen_opto_all_subset_resps.h5",
     "pupil_responses/fig02_opto_subset_A_pupil_responses.h5"),
    ("fig2_musc_A_pre_by_cond.h5",
     "pupil_responses/fig02_muscimol_A_prestim_by_condition.h5"),
    ("fig2_musc_X_pre_by_cond.h5",
     "pupil_responses/fig02_muscimol_X_prestim_by_condition.h5"),
    ("fig2_opto_A_pre_by_cond.h5",
     "pupil_responses/fig02_opto_A_prestim_by_condition.h5"),
    ("fig2_opto_X_pre_by_cond.h5",
     "pupil_responses/fig02_opto_X_prestim_by_condition.h5"),
    ("fig2_dev_opto_A_by_cond_pre.h5",
     "pupil_responses/fig02_opto_deviant_A_prestim_by_condition.h5"),
    ("fig2_dev_opto_X_by_cond_pre.h5",
     "pupil_responses/fig02_opto_deviant_X_prestim_by_condition.h5"),
    ("fig4_opto_A_pre_by_cond.h5",
     "pupil_responses/fig04_opto_A_prestim_by_condition.h5"),
    ("fig4_opto_X_pre_by_cond.h5",
     "pupil_responses/fig04_opto_X_prestim_by_condition.h5"),

    # Figure 3
    (r"D:\ephys\rare_freq_by_pip_resps_ephys_2401_2504.joblib",
     "ephys_responses/fig03_rare_frequency_by_pip_responses_ephys_2401_2504.joblib"),
    (r"D:\ephys\normdev_resps_ephys_2401_2504.joblib",
     "ephys_responses/fig03_normal_deviant_responses_ephys_2401_2504.joblib"),
    (r"X:\Dammy\figures\decoding_plots_aggr_sessions_ephys_2401_2504\stim_decoding_df.h5",
     "decoding/fig03_stimulus_decoding_results.h5"),
    (r"X:\Dammy\figures\decoding_plots_aggr_sessions_ephys_2401_2504\all_v_all_decoding_cm.npy",
     "decoding/fig03_all_pips_confusion_matrices.npy"),
    (r"X:\Dammy\figures\decoding_plots_aggr_sessions_ephys_2401_2504\all_v_all_decoding_cm_shuffle.npy",
     "decoding/fig03_all_pips_confusion_matrices_shuffled.npy"),
    (r"X:\Dammy\figures\decoding_plots_aggr_sessions_ephys_2401_2504\all_v_all_decoding_df.h5",
     "decoding/fig03_all_pips_decoding_results.h5"),
    (r"X:\Dammy\figures\decoding_plots_aggr_sessions_ephys_2401_2504\all_v_all_train_rare_decoding_cm.npy",
     "decoding/fig03_rare_trained_confusion_matrices.npy"),
    (r"X:\Dammy\figures\decoding_plots_aggr_sessions_ephys_2401_2504\all_v_all_train_rare_decoding_cm_shuffle.npy",
     "decoding/fig03_rare_trained_confusion_matrices_shuffled.npy"),
    (r"X:\Dammy\figures\decoding_plots_aggr_sessions_ephys_2401_2504\all_v_all_train_rare_decoding_df.h5",
     "decoding/fig03_rare_trained_decoding_results.h5"),
    (r"X:\Dammy\figures\decoding_plots_aggr_sessions_ephys_2401_2504\rare_freq_decoding_using_post0_5_no_zscore_df.h5",
     "decoding/fig03_rare_frequency_poststim_decoding_results.h5"),
    (r"X:\Dammy\figures\decoding_plots_aggr_sessions_ephys_2401_2504\normdev_decoding_0.5_1.5_df.h5",
     "decoding/fig03_normal_deviant_decoding_results.h5"),
    (r"X:\Dammy\aggr_savedir\trial_stim_responses.h5",
     "aggregate_cache/fig03_stimulus_response_aggregate.h5"),
    (r"X:\Dammy\aggr_savedir\trial_stim_responses_sessnames.npy",
     "aggregate_cache/fig03_stimulus_response_session_names.npy"),

    # Figure 4 aggregate cache
    (r"X:\Dammy\aggr_savedir\A-0_rare_A-0_frequent_early_late_sliding_block_2_only.h5",
     "aggregate_cache/fig04_rare_frequency_response_aggregate.h5"),
    (r"X:\Dammy\aggr_savedir\A-0_rare_A-0_frequent_early_late_sliding_block_2_only_sessnames.npy",
     "aggregate_cache/fig04_rare_frequency_response_session_names.npy"),

    # Figure 5
    (r"D:\ephys\ABCD_ABBA_no_zscore_ephys_2401_2504.joblib",
     "ephys_responses/fig05_abcd_vs_abba_responses_ephys_2401_2504.joblib"),
    (r"X:\Dammy\aggr_savedir_261120\A-0_A-1_A-2_trial_normed_trimmed_A-0.h5",
     "aggregate_cache/fig05a_trial_normed_response_aggregate.h5"),
    (r"X:\Dammy\aggr_savedir_261120\A-0_A-1_A-2_trial_normed_trimmed_A-0_sessnames.npy",
     "aggregate_cache/fig05a_trial_normed_response_session_names.npy"),
    (r"X:\Dammy\aggr_savedir\abcd_vs_abba_means.h5",
     "aggregate_cache/fig05b_abcd_vs_abba_response_aggregate.h5"),
    (r"X:\Dammy\aggr_savedir\abcd_vs_abba_means_sessnames.npy",
     "aggregate_cache/fig05b_abcd_vs_abba_session_names.npy"),

    # Figure configurations and session metadata
    (r"H:\ephys\psth_plot_config_ephys_2401_2504.yaml",
     "configs/fig03_ephys_2401_2504.yaml"),
    (r"H:\ephys\abstraction_plot_config.yaml",
     "configs/fig05a_abstraction.yaml"),
    (r"H:\ephys\abcd_vs_abba_config.yaml",
     "configs/fig05b_abcd_vs_abba.yaml"),
    (r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2401.csv",
     "session_metadata/ephys_session_topology_2401.csv"),
    (r"X:\Dammy\Xdetection_mouse_hf_test\session_topology_ephys_2504.csv",
     "session_metadata/ephys_session_topology_2504.csv"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy figure input files into the paper data directory."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files that already exist in the destination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned copies without copying files.",
    )
    args = parser.parse_args()

    copied = 0
    skipped = 0
    missing: list[Path] = []

    for source_name, destination_name in FILES:
        source = Path(source_name)
        if not source.is_absolute():
            source = SOURCE_ROOT / source
        destination = DESTINATION_ROOT / Path(destination_name)

        if not source.is_file():
            print(f"MISSING  {source}")
            missing.append(source)
            continue

        if destination.exists() and not args.overwrite:
            print(f"EXISTS   {destination} (not overwritten)")
            skipped += 1
            continue

        if args.dry_run:
            print(f"WOULD COPY  {source} -> {destination}")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"COPIED   {source} -> {destination}")
        copied += 1

    print(
        f"\nSummary: {copied} copied, {skipped} already existed, "
        f"{len(missing)} missing."
    )
    if missing:
        print("Review the MISSING paths above; no substitute source paths were used.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
