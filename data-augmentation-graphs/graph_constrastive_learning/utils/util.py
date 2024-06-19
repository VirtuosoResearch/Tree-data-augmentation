import pandas as pd
import json
import logging
import logging.config
import os
from collections import OrderedDict
from pathlib import Path
from datetime import datetime

os.environ['NUMEXPR_MAX_THREADS'] = '16'

def add_result_to_csv(result_datapoint, file_name):
    for key, val in result_datapoint.items():
        result_datapoint[key] = [val, ]
    
    if os.path.exists(file_name):
        result_df = pd.read_csv(file_name, index_col=0)
        tmp_df = pd.DataFrame(result_datapoint)
        result_df = pd.concat([result_df, tmp_df], ignore_index = True)
        result_df.to_csv(file_name)
    else:
        result_df = pd.DataFrame(result_datapoint)  
        result_df.to_csv(file_name)   

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)
    
def setup_logging_logic():
    # set up log file
    log_dir = "./saved/logs/{}".format(datetime.now().strftime(r'%m%d_%H%M%S'))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    setup_logging(log_dir)

def setup_logging(save_dir, log_config='utils/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(Path(save_dir) / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)

def get_logger(name, verbosity=2):
    log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, log_levels.keys())
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(name)
    logger.setLevel(log_levels[verbosity])
    return logger

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
