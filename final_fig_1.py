import json
import platform
from pathlib import Path
import sys

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pupil_analysis_funcs import PupilCondAnalysis, run_pupil_cond_analysis
from save_utils import save_stats_to_tex
from xdetectioncore.io_utils import load_response_dfs
from xdetectioncore.behaviour import get_all_cond_filts
from xdetectioncore.plotting import format_axis

if __name__ == '__main__':

    plt.style.use('figure_stylesheet.mplstyle')

    # Load pre-computed condition response dictionaries
    A_by_cond_path = Path(r"X:\Xdetection_data\data_for_paper\fig1_A_by_cond.h5")
    X_by_cond_path = Path(r"X:\Xdetection_data\data_for_paper\fig1_X_by_cond.h5")

    A_by_cond = load_response_dfs(A_by_cond_path)
    X_by_cond = load_response_dfs(X_by_cond_path)

    # Load trial data
    all_td_df_path = Path(r"X:\Xdetection_data\data_for_paper\all_td_df.h5")
    all_td_df = pd.read_hdf(all_td_df_path)
    assert isinstance(all_td_df, pd.DataFrame)

    cond_filters = get_all_cond_filts()
    good_trial_filt = 'n_since_last_Trial_Outcome <=5 '

    # Load session lists by condition
    sesslist_path = Path('sesslist_by_cond.json')
    if sesslist_path.is_file():
        with open(sesslist_path, 'r') as f:
            sesslist_by_cond = json.load(f)
        all_rare_freq_sess = sesslist_by_cond['event_rate']
        norm_dev_sesslist = sesslist_by_cond['norm_dev']
        dev_sesslist = sesslist_by_cond['generalisation']
    else:
        # Fallback to unique sessions found across keys
        all_rare_freq_sess = list(A_by_cond['rare'].index.get_level_values('sess').unique())
        norm_dev_sesslist = list(A_by_cond['normal'].index.get_level_values('sess').unique())
        dev_sesslist = list(A_by_cond['dev_ABCD1'].index.get_level_values('sess').unique())

    # Common styling kwargs
    boxplot_kwargs = dict(
        widths=0.5,
        patch_artist=False,
        showmeans=False,
        showfliers=False,
        medianprops=dict(mfc='k'),
        boxprops=dict(lw=0.5),
        whiskerprops=dict(lw=0.5),
        capprops=dict(lw=0.5),
    )

    xlim = [-0.25, 2.25]

    # =========================================================================
    # 1. X Response Plot (hit_all vs miss_all)
    # =========================================================================
    x_sess_list = X_by_cond['miss_all'].index.get_level_values('sess').unique()
    X_resp = PupilCondAnalysis(
        by_cond_dict=X_by_cond,
        sess_list=x_sess_list,
        conditions=['hit_all', 'miss_all'],
        event_name='X'
    )
    x_line_kwargs = {
        'hit_all': {'c': '#31572c', 'ls': '-'},
        'miss_all': {'c': '#31572c', 'ls': '--'}
    }
    X_resp.plot_ts_by_cond(cond_line_kwargs=x_line_kwargs)
    X_resp.plots['ts_by_cond'][0].set_size_inches(1.5, 2)
    X_resp.plots['ts_by_cond'][0].show()
    X_resp.plots['ts_by_cond'][0].savefig('X_resp.pdf')

    # =========================================================================
    # 2. X Intermediate Responses Plot (AB, ABC, ABCD)
    # =========================================================================
    if all(cond in X_by_cond for cond in ['AB', 'ABC', 'ABCD']):
        x_inter_resps = PupilCondAnalysis(
            by_cond_dict=X_by_cond,
            conditions=['AB', 'ABC', 'ABCD'],
            event_name='X'
        )
        early_x_line_kwargs = {
            'AB': {'c': '#31572c', 'ls': '-'},
            'ABC': {'c': '#84a98c', 'ls': '-'},
            'ABCD': {'c': '#c2c5aa', 'ls': '-'}
        }
        x_inter_resps.plot_ts_by_cond(cond_line_kwargs=early_x_line_kwargs)
        x_inter_resps.plots['ts_by_cond'][0].set_size_inches(2, 2)
        x_inter_resps.plots['ts_by_cond'][0].show()
        x_inter_resps.plots['ts_by_cond'][0].savefig('x_inter_resp.pdf')

    # =========================================================================
    # 3. Rare vs Frequent Analysis
    # =========================================================================

    rare_freq_line_kwargs = {
        'rare': {'c': '#00cbfcff', 'ls': '-'},
        'frequent': {'c': '#1530e9ff', 'ls': '-'}
    }
    rare_freq_condnames = ['frequent', 'rare']

    for stim_name, stim_by_cond in zip(['A', 'X'], [A_by_cond, X_by_cond]):
        rare_freq_analysis = PupilCondAnalysis(stim_by_cond, rare_freq_condnames,all_rare_freq_sess)
        rare_freq_analysis.plot_ts_by_cond(cond_line_kwargs=rare_freq_line_kwargs)
        rare_freq_analysis.plots['ts_by_cond'][1].set_xlim(xlim)
        rare_freq_analysis.plots['ts_by_cond'][0].set_size_inches(2, 2)
        rare_freq_analysis.plots['ts_by_cond'][0].show()

    # Behavior: Rare vs Frequent
    cond_td_df = all_td_df.loc[all_td_df.index.get_level_values('sess').isin(all_rare_freq_sess)]
    rare_freq_perf = [
        cond_td_df.query(good_trial_filt).query(cond_filters[cond])['Trial_Outcome'].groupby('name').mean()
        for cond in rare_freq_condnames
    ]
    fig, ax = plt.subplots()
    ax.bar(rare_freq_condnames, [e.mean() for e in rare_freq_perf], ec='black', fc='w', lw=0.5, width=0.4)
    for cond, data in zip(rare_freq_condnames, rare_freq_perf):
        ax.scatter([cond] * len(data), data, edgecolor='none', fc=rare_freq_line_kwargs[cond]['c'], alpha=0.5)

    format_axis(ax)
    fig.set_size_inches(2, 2)
    fig.set_layout_engine('constrained')
    fig.show()

    # =========================================================================
    # 4. Normal vs Deviant Analysis
    # =========================================================================

    norm_dev_line_kwargs = {
        'normal': {'c': 'black', 'ls': '-'},
        'deviant_C': {'c': '#cd2727ff', 'ls': '-'}
    }
    norm_dev_cond_names = ['normal', 'deviant_C']

    for stim_name, stim_by_cond in zip(['A', 'X'], [A_by_cond, X_by_cond]):
        norm_dev_analysis = PupilCondAnalysis(stim_by_cond, norm_dev_cond_names,norm_dev_sesslist)
        norm_dev_analysis.plot_ts_by_cond(cond_line_kwargs=norm_dev_line_kwargs)
        norm_dev_analysis.plots['ts_by_cond'][1].set_xlim(xlim)
        norm_dev_analysis.plots['ts_by_cond'][0].set_size_inches(2, 2)
        norm_dev_analysis.plots['ts_by_cond'][0].show()

    # Behavior: Normal vs Deviant
    cond_td_df = all_td_df.loc[all_td_df.index.get_level_values('sess').isin(norm_dev_sesslist)]
    norm_dev_perf = [
        cond_td_df.query(good_trial_filt).query(cond_filters[cond])['Trial_Outcome'].groupby('name').mean()
        for cond in norm_dev_cond_names
    ]

    fig, ax = plt.subplots()
    ax.bar(norm_dev_cond_names, [e.mean() for e in norm_dev_perf], ec='black', fc='w', lw=0.5, width=0.4)
    for cond, data in zip(norm_dev_cond_names, norm_dev_perf):
        ax.scatter([cond] * len(data), data, edgecolor='none', fc=norm_dev_line_kwargs[cond]['c'], alpha=0.5)

    format_axis(ax)
    fig.set_size_inches(1.5, 2)
    fig.set_layout_engine('constrained')
    fig.show()

    # =========================================================================
    # 5. dev_ABCD1 and dev_ABBA1 Analysis
    # =========================================================================

    dev_cond_names = ['normal', 'dev_ABCD1', 'dev_ABBA1']
    dev_line_kwargs = {
        'normal': {'c': 'k', 'ls': '-'},
        'dev_ABCD1': {'c': '#e8739bff', 'ls': '-'},
        'dev_ABBA1': {'c': '#73e8c1ff', 'ls': '-'}
    }
    dev_ABBA1_ABCD_analysis = PupilCondAnalysis(A_by_cond, dev_cond_names,dev_sesslist)
    dev_ABBA1_ABCD_analysis.plot_ts_by_cond(cond_line_kwargs=dev_line_kwargs)
    dev_ABBA1_ABCD_analysis.plots['ts_by_cond'][1].set_xlim(xlim)
    dev_ABBA1_ABCD_analysis.plots['ts_by_cond'][0].show()

