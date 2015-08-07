from source_py.data import DataLoader
from source_py.model import ContentFilter

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

        display = loader.coupons_train.DISPFROM

        # use coupons in training period for training
        self.train = loader.coupons_train.copy(deep=True)
        self.train.drop(loader.coupons_train[display >= self.end].index, inplace=True)
        self.train.drop(loader.coupons_train[display < self.start].index, inplace=True)

        # use coupons in validation period for testing
        self.test = loader.coupons_train.copy(deep=True)
        self.test.drop(loader.coupons_train[display >= self.valid_end].index, inplace=True)
        self.test.drop(loader.coupons_train[display < self.end].index, inplace=True)

        trans = loader.details_train

        # only allow transaction history of coupons in the training set within training period
        self.purchases = trans[trans.COUPON_ID_hash.isin(self.train.COUPON_ID_hash)].copy(deep=True)
        self.purchases.drop(self.purchases[self.purchases.I_DATE >= self.end].index, inplace=True)

        # actual purchases made during validation period
        tp = trans.copy(deep=True)
        tp.drop(trans[trans.I_DATE >= self.valid_end].index, inplace=True)
        tp.drop(trans[trans.I_DATE < self.end].index, inplace=True)
        tp.drop(tp[tp.COUPON_ID_hash.isin(self.train.COUPON_ID_hash)].index, inplace=True)
        self.test_purchases = tp

        # check that all test purchases come from test coupon data
        assert tp[tp.COUPON_ID_hash.isin(self.test.COUPON_ID_hash)].shape == tp.shape

        # users
        self.users = loader.user_list


    def run(self, mode="content_filter"):

        if mode == "content_filter":
            model = ContentFilter(self.train, self.test, self.users, self.purchases)
            model.run()
            predictions = model.predict()
            self.MAP(predictions)
        else:
            raise NotImplementedError

    def MAP(self, predictions):
        """
        :param predictions: pandas.DataFrame of predictions in Kaggle format
        :return: Mean Average Precision using actual purchases in validation period
        """
        pass


