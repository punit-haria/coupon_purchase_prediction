from source_py.data import DataLoader

import numpy as np

class Validator(object):

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

        #TODO (See below)
        '''
        Cross validate by taking all coupons with a DISPFROM value on or before
        the end training date and put them in the training set. Coupons with a DISPFROM
        value between the training end date and the end of the validation period go
        in the validation set. Similarly, split the purchases DataFrame by joining on
        these newly created coupon training and validation sets.
        '''
