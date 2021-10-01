from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import re
from datetime import datetime

# Regular Expression to extract time stamp from image name
PATTERN = re.compile('.*_(\d*)_(\d*)_(\d*)_(\d*)_(\d*)_(\d*)_(\d*)\.jpg')
COL_NAMES = [
            'center_img',
            'left_img',
            'right_img', 
            'steering_angle', 
            'throttle', 
            'brake', 
            'speed'
            ]

def match_to_datestamp(match: re.Match) -> datetime:
    """Helper function to convert match to a datetime object
    |   match object must have 7 groups where 
    |   group 0 is year
    |   group 1 is month
    |   group 2 is day
    |   group 3 is hour
    |   group 4 is minute
    |   group 5 is second
    |   group 6 is millisecond"""
    groups = [int(elm) for elm in match.groups()]
    try:
        assert len(groups) == 7
    except AssertionError:
        print("Match object may be in wrong format check the regular expression used.")
        raise AssertionError
    date = groups[:3] 
    time = groups[3:-1]
    microseconds = groups[-1]*1000 #convert millisecond to microsecond for datetime
    return datetime(date[0], date[1], date[2], hour = time[0], 
        minute = time[1], second = time[2], microsecond=microseconds)

def string_to_datestamp(mystring : str, pattern : re.Pattern = PATTERN) -> datetime:
    """Convert img pathnames to datetime"""
    match = pattern.match(mystring)
    date = match_to_datestamp(match)
    return date

def find_records(df:pd.DataFrame, threshold: np.timedelta64 = np.timedelta64(100_000_000)) -> list[pd.DataFrame]:
    if not 'timestamps' in df.columns:
        _df = add_timestamps_to_dataframe(df, inplace=False)
    else:
        _df = df.copy()
    time_deltas = _df['timestamps'].diff()
    breaks = _df[time_deltas>threshold].index
    records = []
    prev_ind = 0
    for current_ind in breaks:
        records.append(_df.iloc[prev_ind:current_ind-1])
        prev_ind=current_ind
    records.append(_df.iloc[prev_ind:])

    return records

def add_timestamps_to_dataframe(df:pd.DataFrame, inplace:bool = False) -> Union[None, pd.DataFrame]:
    
    if not inplace:
        _df = df.copy()
    else:
        _df = df
    
    _df['timestamps'] = _df['center_img'].map(string_to_datestamp)
    
    if not inplace:
        return _df

def import_driving_log(path_to_driving_log: Union[Path, str], data_path: str) -> pd.DataFrame:
    
    df = pd.read_csv(path_to_driving_log, 
        header=None,
        names=COL_NAMES)
    def _correct_data_paths(split_path : list[str]) -> str:
        """Helper function of import_driving_log converts original data path to correct data path"""
        img_path = '/'.join(split_path[-3:])
        if data_path[-1] == "/":
            return data_path + img_path
        else:
            return data_path + "/" + img_path
    
    for col in COL_NAMES[:3]:
        df[col] = df[col].str.split('\\').map(_correct_data_paths)
    
    return df

