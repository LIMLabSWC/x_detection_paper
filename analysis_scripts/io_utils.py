import json
import pickle
import platform
import re
from datetime import datetime
from pathlib import Path, PureWindowsPath, PurePosixPath

import joblib
import numpy as np
import pandas as pd
import h5py


def format_sound_writes(sound_writes_df: pd.DataFrame, patterns: list[int, ],normal_patterns=None,
                        ptypes=None ) -> pd.DataFrame:
    # assign sounds to trials
    sound_writes_df = sound_writes_df.drop_duplicates(subset='Timestamp', keep='first').copy()
    sound_writes_df = sound_writes_df.reset_index(drop=True)
    sound_writes_df['rand_n'] = np.random.randint(0,8,len(sound_writes_df)).astype(int)
    # sound_writes_df['Timestamp'] = sound_writes_df['Timestamp']
    sound_writes_df['Trial_Number'] = np.full_like(sound_writes_df.index, -1)
    sound_writes_diff = sound_writes_df['Time_diff'] = sound_writes_df['Timestamp'].diff()
    # print(sound_writes_diff[:10])
    if 3 in sound_writes_df['Payload'].values:
        long_dt = np.squeeze(np.argwhere(sound_writes_diff.values > 2))
    else:
        long_dt = np.squeeze(np.argwhere(sound_writes_diff.values > 1))

    for n, idx in enumerate(long_dt):
        sound_writes_df.loc[idx:, 'Trial_Number'] = n
    sound_writes_df['Trial_Number'] = sound_writes_df['Trial_Number'] + 1

    sound_writes_df['Time_diff'] = sound_writes_df['Timestamp'].diff()
    sound_writes_df['Payload_diff'] = sound_writes_df['Payload'].diff()
    if 3 in sound_writes_df['Payload'].values:
        matrix_d_X_times = np.array(np.matrix(sound_writes_df['Timestamp'].values).T -
                                    sound_writes_df.query('Payload == 3')['Timestamp'].values)
        matrix_d_X_times[matrix_d_X_times > 0] = 9999
        matrix_d_X_times = np.min(np.abs(matrix_d_X_times),axis=1)
    else:
        matrix_d_X_times = np.full_like(sound_writes_df['Timestamp'].values, np.nan)
    sound_writes_df['d_X_times'] = matrix_d_X_times

    tones = sound_writes_df['Payload'].unique()
    pattern_tones = tones[tones >= 8]
    if patterns is not None:
        base_pip_idx = [e for e in pattern_tones if e not in sum(patterns, [])]
        if len(base_pip_idx) > 1:
            base_pip_idx = sound_writes_df.query('Payload in @base_pip_idx')['Payload'].mode().iloc[0]
        elif len(base_pip_idx) == 1:
            base_pip_idx = base_pip_idx[0]
        else:
            raise ValueError(f'No base_pip_idx found. tones: {tones}')
    else:
        base_pip_idx = sound_writes_df['Payload'].mode().iloc[0]

    if patterns is None or 3 not in sound_writes_df['Payload'].values:
        normal_patterns, deviant_patterns = None, None
        sound_writes_df['pattern_pips'] = [np.nan]*sound_writes_df.shape[0]
        sound_writes_df['pattern_start'] = [np.nan]*sound_writes_df.shape[0]
        sound_writes_df['pip_counter'] = [np.nan]*sound_writes_df.shape[0]
        sound_writes_df['ptype'] = [np.nan]*sound_writes_df.shape[0]
    else:
        if normal_patterns is None:
            normal_patterns = [pattern for pattern in patterns if (np.diff(pattern) > 0).all()]
        deviant_patterns = [pattern for pattern in patterns if pattern not in normal_patterns]

        sound_writes_df['pattern_pips'] = sound_writes_df['Payload'].isin(sum(patterns, []))
        sound_writes_df['pattern_start'] = sound_writes_df['Payload_diff'].isin(np.array(patterns)[:, 0] - base_pip_idx) * \
                                           sound_writes_df['pattern_pips']
        sound_writes_df['pip_counter'] = np.zeros_like(sound_writes_df['pattern_start']).astype(int)
        patterns_presentations = [sound_writes_df.iloc[idx:idx+4]
                                  for idx in sound_writes_df.query('pattern_start == True').index
                                  if base_pip_idx not in sound_writes_df.iloc[idx:idx+4]['Payload'].values
                                  and 3 not in sound_writes_df.iloc[idx:idx+4]['Payload'].values]
        sound_writes_df['ptype'] = [np.nan]*sound_writes_df.shape[0]
        for pattern_pres in patterns_presentations:
            patt = pattern_pres['Payload'].to_list()
            if ptypes is not None:
                ptype = ptypes['_'.join(str(e) for e in patt)]
            else:
                ptype = 0 if patt in normal_patterns else 1 if patt in deviant_patterns else -1
            sound_writes_df.loc[pattern_pres.index, 'ptype'] = ptype

        for i in sound_writes_df.query('pattern_start == True').index:
            sound_writes_df.loc[i:i + 3, 'pip_counter'] = np.cumsum(sound_writes_df['pattern_pips'].loc[i:i + 3]) \
                                                          * sound_writes_df['pattern_pips'].loc[i:i + 3]
    return sound_writes_df

def read_lick_times(beh_path):
    beh_events_df = pd.read_csv(beh_path)
    lick_times = beh_events_df[beh_events_df['Payload'] == 0]['Timestamp']
    return lick_times.values


def posix_from_win(path: str, ceph_linux_dir='/ceph/akrami') -> Path:
    """
    Convert a Windows path to a Posix path.

    Args:
        path (str): The input Windows path.
        :param ceph_linux_dir:

    Returns:
        Path: The converted Posix path.
    """
    if ':\\' in path:
        path_bits = PureWindowsPath(path).parts
        path_bits = [bit for bit in path_bits if '\\' not in bit]
        return Path(PurePosixPath(*path_bits))
    else:
        assert ceph_linux_dir
        return Path(path).relative_to(ceph_linux_dir)


def split_path_cross_platform(path_str):
    # Normalize all separators to backslash for detection
    norm_path = path_str.replace('/', '\\')

    # Detect Windows style:
    # - Starts with drive letter: e.g., C:\ or D:/
    # - Starts with UNC path: \\server\share
    is_windows = bool(
        re.match(r'^[a-zA-Z]:\\', norm_path) or
        norm_path.startswith('\\\\')
    )

    if is_windows:
        return PureWindowsPath(path_str).parts
    else:
        return PurePosixPath(path_str).parts


def get_spikedir(recdir, sorter='kilosort2_5', sorting_dir_name='sorting') -> Path:
    """
    Returns the path to the spike sorting output directory for the given recording directory and sorter.

    Parameters:
    recdir (Path): The path to the recording directory.
    sorter (str): The spike sorter name (default is 'kilosort2_5').

    Returns:
    Path: The path to the spike sorting output directory.
    """
    recdir = Path(recdir)

    while not any([sorting_dir_name in str(path) for path in list(recdir.iterdir())]):
        recdir = recdir.parent

    spikedir = recdir / sorting_dir_name / sorter / 'sorter_output'
    assert spikedir.is_dir(), Warning(f'spikedir {spikedir} not a dir, check path and sorter name')

    return spikedir


def load_spikes(spike_times_path: [Path | str], spike_clusters_path: [Path | str], parent_dir=None):
    """
    A function to load spike times and clusters from the given paths.

    :param spike_times_path: Path to the spike times file
    :param spike_clusters_path: Path to the spike clusters file
    :param parent_dir: Optional parent directory for the spike times and clusters paths
    :return: Tuple containing spike times and spike clusters arrays
    """
    spike_times_path, spike_clusters_path = Path(spike_times_path), Path(spike_clusters_path)
    assert spike_times_path.suffix == '.npy' and spike_clusters_path.suffix == '.npy', Warning('paths should be .npy')

    if parent_dir:
        print(f'parent_dir = {parent_dir}')
        spike_times_path, spike_clusters_path = [next(parent_dir.rglob(path.name)) or path
                                                 for path in [spike_times_path, spike_clusters_path]
                                                 if not path.is_absolute()]
    spike_times = np.load(spike_times_path)
    spike_clusters = np.load(spike_clusters_path)

    return spike_times, spike_clusters


def load_pupil_data(pupil_data_path: Path | str, parent_dir=None,sess_list=None):
    """
    A function to load pupil data from the given path.

    :param pupil_data_path: Path to the pupil data file
    :param parent_dir: Optional parent directory for the pupil data path
    :return: Pupil data array
    """
    pupil_data_path = Path(pupil_data_path) if isinstance(pupil_data_path,str) else pupil_data_path
    # pupil_data = pickle.load(f)
    pupil_data = {}
    # while True:
    #     try:
    #         y = joblib.load(pupil_data_path)
    #         y = {k: y[k] for k in y.keys() if y[k].pupildf is not None}
    #         z = {**pupil_data, **y}
    #         pupil_data = z
    #     except EOFError:
    #         print(f'end of file {pupil_data.keys()}')
    #         break
    pupil_data = joblib.load(pupil_data_path)
    for sess in list(pupil_data.keys()):
        if pupil_data[sess].pupildf is None:
            pupil_data.pop(sess)
        else:
            pupil_data[sess].pupildf.index = pupil_data[sess].pupildf.index
        if sess_list:
            name = sess.split('_')[0]
            date = extract_date(sess)
            if not any(['_'.join([name,date]) in s for s in sess_list]) and sess in pupil_data.keys():
                pupil_data.pop(sess)
    return pupil_data


def load_sound_bin(binpath: Path):
    # all_writes = pd.read_csv(bin_dir/f'{bin_stem}_write_indices.csv')
    all_writes = pd.read_csv(binpath)
    return all_writes






def load_sess_pkl(pkl_path):
    sys_os = platform.system().lower()
    if sys_os == 'windows':
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
    if sys_os == 'linux':
        import pathlib
        temp = pathlib.WindowsPath
        pathlib.WindowsPath = pathlib.PosixPath
    with open(pkl_path, 'rb') as pklfile:
        try:
            pklfile.seek(0)
            sess_obj = pickle.load(pklfile)
        except AttributeError:
            print(f'{pkl_path} AttributeError. Likely created before refactoring')
            return None
        except EOFError:
            print(f'{pkl_path} pickle error. Check pickle')
            return None
        # sess_obj.sound_event_dict = {}
        # sessions[sess_obj.sessname] = sess_obj
    return sess_obj


def sync_using_latest_ttl(sink_events, sink_ttls, source_ttls):
    """
    Sync sink-side event times to source time using the most recent TTL only.

    :param sink_events: np.array of sink-side event times to sync
    :param sink_ttls: np.array of sink TTL times (must be sorted)
    :param source_ttls: np.array of corresponding source TTL times (same length)
    :return: np.array of synced source times
    """
    # Find index of the closest preceding sink TTL for each event
    idxs = np.searchsorted(sink_ttls, sink_events, ) - 1
    idxs = np.clip(idxs, 0, len(sink_ttls) - 1)  # Handle early events

    # Time offset between clocks at each TTL
    offset = sink_ttls[idxs] - source_ttls[idxs]

    print(f'1 s in source TTL clock = {(sink_ttls[-1]-sink_ttls[0])/(source_ttls[-1]-source_ttls[0]):.6f} s')
    # Apply correction
    return sink_events - offset


def extract_date(filename_str:str,out_fmt:str='%y%m%d'):
    try:
        date = re.search(r'\d{4}-\d{2}-\d{2}', filename_str).group(0)
        date = datetime.strptime(date, '%Y-%m-%d').strftime('%y%m%d')
    except AttributeError:
        date = re.search(r'\d{2}\d{2}\d{2}', filename_str).group(0)
    return datetime.strptime(date, '%y%m%d').strftime(out_fmt)


def load_and_map_json(file_path):
    """
    Loads the JSON and transforms the structure so you can access data via:
    data['Key']['ColumnName'] -> [value1, value2, ...]
    """
    try:
        with open(file_path, 'r') as f:
            raw_data = json.load(f)

        refined_data = {}

        for category, content in raw_data.items():
            # Create a sub-dictionary for each category (e.g., 'PWM', 'LED0')
            refined_data[category] = {}

            # Extract mapping and columns
            lookup = content.get('colindex', {}).get('lookup', {})
            columns = content.get('columns', [])

            for col_name, one_based_idx in lookup.items():
                # 1. Convert 1-based index to 0-based
                zero_idx = one_based_idx - 1

                # 2. Map the column name directly to its data list
                # We use columns[zero_idx] to flatten the nested [[...]] structure
                if zero_idx < len(columns):
                    refined_data[category][col_name] = columns[zero_idx]
                else:
                    refined_data[category][col_name] = []

        return refined_data

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return None

def save_dict_to_h5(data_dict: dict, h5_path: Path):
    """
    Modular function to recursively save a dictionary of NumPy arrays
    (or nested dictionaries) into an HDF5 file.
    """
    with h5py.File(h5_path, 'w') as f:
        def _recursive_save(group, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    sub_grp = group.create_group(str(k))
                    _recursive_save(sub_grp, v)
                elif isinstance(v, np.ndarray):
                    group.create_dataset(str(k), data=v)
                elif isinstance(v, (list, tuple)):
                    group.create_dataset(str(k), data=np.array(v))
                else:
                    group.attrs[str(k)] = str(v)
        _recursive_save(f, data_dict)

def load_dict_from_h5(h5_path: Path) -> dict:
    """
    Modular function to recursively load an HDF5 file back
    into a dictionary of NumPy arrays.
    """
    with h5py.File(h5_path, 'r') as f:
        def _recursive_load(group):
            res = {}
            for k, item in group.items():
                if isinstance(item, h5py.Group):
                    res[k] = _recursive_load(item)
                elif isinstance(item, h5py.Dataset):
                    res[k] = item[()]
            for k, v in group.attrs.items():
                res[k] = v
            return res
        return _recursive_load(f)