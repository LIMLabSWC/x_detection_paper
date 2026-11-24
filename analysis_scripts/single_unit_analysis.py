import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.rcsetup import cycler

from plot_funcs import plot_shaded_error_ts

# matplotlib.use('TkAgg')


def plot_unit_resp(unit_resp_plot,unit_resp_dict,sessnames,unit_sem_dict,x_ser, n_units=None,idx_mask=None,units=None):
    if idx_mask is None:
        idx_mask = np.ones(len(sessnames))

    if units is None:
        units = np.arange(len(sessnames))
    for ui, ax in enumerate(unit_resp_plot[1].flatten()):
        if ui >= len(sessnames[idx_mask]) or ui >= n_units:
            continue
        u_idx = units[ui]
        [ax.plot(x_ser, resp[idx_mask][u_idx, :], label=lbl.split('_')[-1]) for lbl, resp in unit_resp_dict.items()]
        [plot_shaded_error_ts(ax, x_ser, mean_ts[idx_mask][u_idx, :], sem_ts[idx_mask][u_idx, :], alpha=0.1)
         for mean_ts, sem_ts in zip(unit_resp_dict.values(), unit_sem_dict.values())]
        ax.set_title(f'{sessnames[idx_mask][u_idx]}: {u_idx}', fontsize=7)


if __name__ == '__main__':
    ### rare freq ###
    rare_freq_resp_paths = [r"X:\Dammy\aggr_savedir\A-0_rare_A-0_frequent_smooth.joblib",
                            r"X:\Dammy\aggr_savedir\A-0_rare_A-0_frequent_smooth_sem.joblib",
                            r"X:\Dammy\aggr_savedir\A-0_rare_A-0_frequent_sessnames.joblib"]
    rare_freq_resps, rare_freq_resps_sem,rare_freq_sessnames = [joblib.load(p) for p in rare_freq_resp_paths]
    plt.style.use('figure_stylesheet.mplstyle')

    rare_freq_line_kwargs = {
        'rare': {'c': '#00cbfcff', 'ls': '-'},
        'frequent': {'c': '#1530e9ff', 'ls': '-'}
    }
    plt.rcParams['axes.prop_cycle'] = cycler(color=[v['c'] for v in rare_freq_line_kwargs.values()])


    idxs_2_use = ['DO97' in sess for sess in rare_freq_sessnames]

    x_ser = np.round(np.arange(-0.5,1.51,0.01),2)

    n_units = 100
    unit_resp_plot = plt.subplots(nrows=int(n_units/10),ncols=10,figsize=(20,20*n_units/100))
    plot_unit_resp(unit_resp_plot,rare_freq_resps,rare_freq_sessnames,rare_freq_resps_sem,x_ser,n_units,idxs_2_use)
    [ax.axvline(0,ls='--',c='k') for ax in unit_resp_plot[1].flatten()]
    unit_resp_plot[1][0,-1].legend()
    unit_resp_plot[0].set_layout_engine('constrained')
    unit_resp_plot[0].savefig('unit_resp_plot.pdf')

    good_units = [1,3,9,23,26,28][:2]
    rare_freq_examples_plot = plt.subplots(nrows=len(good_units),ncols=1,figsize=(1,1.5),sharex='all')
    plot_unit_resp(rare_freq_examples_plot,rare_freq_resps,rare_freq_sessnames,rare_freq_resps_sem,
                   x_ser,idx_mask=idxs_2_use,units=good_units, n_units=2)
    [ax.axvline(0,ls='--',c='k') for ax in rare_freq_examples_plot[1].flatten()]
    [ax.set_title('') for ax in rare_freq_examples_plot[1].flatten()]
    [ax.set_xlim(-0.1,1.25) for ax in rare_freq_examples_plot[1].flatten()]
    rare_freq_examples_plot[0].set_layout_engine('constrained')
    rare_freq_examples_plot[0].savefig('rare_freq_examples.pdf')
    rare_freq_examples_plot[0].show()

    ## Normal vs deviant
    norm_dev_resp_paths = [r"X:\Dammy\aggr_savedir\A-0_A-1_1_trial_back_smooth.joblib",
                           r"X:\Dammy\aggr_savedir\A-0_A-1_1_trial_back_smooth_sem.joblib",
                           r"X:\Dammy\aggr_savedir\A-0_A-1_1_trial_back_sessnames.joblib"]
    norm_dev_resps, norm_dev_resps_sem,norm_dev_sessnames = [joblib.load(p) for p in norm_dev_resp_paths]

    norm_dev_line_kwargs = {
        'normal': {'c': 'black', 'ls': '-'},
        'deviant_C': {'c': '#cd2727ff', 'ls': '-'}
    }
    plt.rcParams['axes.prop_cycle'] = cycler(color=[v['c'] for v in norm_dev_line_kwargs.values()])

    idxs_2_use = ['DO97' in sess for sess in norm_dev_sessnames]

    x_ser = np.round(np.arange(-0.5,1.51,0.01),2)

    n_units = min(100,len(norm_dev_sessnames[idxs_2_use]))
    norm_dev_unit_resp_plot = plt.subplots(nrows=int(np.ceil(n_units/10)),ncols=10,figsize=(20,20*n_units/100))
    plot_unit_resp(norm_dev_unit_resp_plot,norm_dev_resps,norm_dev_sessnames,norm_dev_resps_sem,x_ser,n_units,idxs_2_use)
    [ax.axvline(0,ls='--',c='k') for ax in norm_dev_unit_resp_plot[1].flatten()]
    [ax.axvline(0.5,ls='--',c='r') for ax in norm_dev_unit_resp_plot[1].flatten()]
    norm_dev_unit_resp_plot[1][0,-1].legend()
    norm_dev_unit_resp_plot[0].set_layout_engine('constrained')
    norm_dev_unit_resp_plot[0].savefig('norm_dev_unit_resp_plot.pdf')

    norm_dev_good_units = [0,2,20][1:] # DO97
    # norm_dev_good_units =  [28,311,405] # DO95
    normdev_examples_plot = plt.subplots(ncols=1,nrows=len(norm_dev_good_units),figsize=(1,1.5),sharex='all')
    plot_unit_resp(normdev_examples_plot,norm_dev_resps,norm_dev_sessnames,norm_dev_resps_sem,
                   x_ser,idx_mask=idxs_2_use,units=norm_dev_good_units, n_units=2)
    [ax.axvline(0,ls='--',c='k') for ax in normdev_examples_plot[1].flatten()]
    [ax.axvline(0.5,ls='--',c='r') for ax in normdev_examples_plot[1].flatten()]
    [ax.set_title('') for ax in normdev_examples_plot[1].flatten()]
    [ax.set_xlim(-0.1,1.25) for ax in normdev_examples_plot[1].flatten()]
    normdev_examples_plot[0].set_layout_engine('constrained')
    normdev_examples_plot[0].savefig('normdev_examples.pdf')
    normdev_examples_plot[0].show()

    ### Dev abstract ###

    dev_abs_resp_paths = [r"X:\Dammy\aggr_savedir\A-0_A-1_A-2_smooth.joblib",
                          r"X:\Dammy\aggr_savedir\A-0_A-1_A-2_smooth_sem.joblib",
                          r"X:\Dammy\aggr_savedir\A-0_A-1_A-2_sessnames.joblib"]
    dev_abs_resps, dev_abs_resps_sem,dev_abs_sessnames = [joblib.load(p) for p in dev_abs_resp_paths]

    plt.rcParams['axes.prop_cycle'] = cycler(color=['k', '#e8739bff', '#73e8c1ff'])

    idxs_2_use = ['DO95' in sess for sess in dev_abs_sessnames]

    x_ser = np.round(np.arange(-0.5,1.51,0.01),2)

    n_units = min(1000,len(dev_abs_sessnames[idxs_2_use]))
    dev_abs_unit_resp_plot = plt.subplots(nrows=int(np.ceil(n_units/10)),ncols=10,figsize=(20,20*n_units/100))
    plot_unit_resp(dev_abs_unit_resp_plot,dev_abs_resps,dev_abs_sessnames,dev_abs_resps_sem,x_ser,n_units,idxs_2_use)
    [ax.axvline(0,ls='--',c='k') for ax in dev_abs_unit_resp_plot[1].flatten()]
    dev_abs_unit_resp_plot[1][0,-1].legend()
    dev_abs_unit_resp_plot[0].set_layout_engine('constrained')
    dev_abs_unit_resp_plot[0].savefig('dev_abs_unit_resp_plot.pdf')