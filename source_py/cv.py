from source_py.data import DataLoader

import numpy as np

class CrossValidator(object):

    def __init__(self, start_date, training_period, validation_period, loader):
        """
        :param start_date: string for start date for training period (format "%Y-%m-%d")
        :param training_period: length of training period in days
        :param validation_period: length of validation period in days
        :param loader: DataLoader
        """
        self.start = np.datetime64(start_date)
        self.end = self.start + training_period
        self.valid_end = self.end + validation_period

        assert isinstance(loader, DataLoader)
        self.data = loader


    