from source_py.data import DataLoader
from source_py.item import ItemProfile
from source_py.user import User

import pandas as pd


class Model(object):

    def __init__(self, loader):
        """
        :param loader: DataLoader
        """
        assert isinstance(loader, DataLoader)
        self.data = loader

    def run(self, start, end):
        """
        :param start: start index in training set
        :param end: end index in training set
        The entire model training process is done using this subset of the training data.
        """
        raise NotImplementedError

    def predict(self, start, end, train=True):
        """
        :param start: start index in DataFrame
        :param end: end index in DataFrame
        Predictions are made for rows in [start:end). If train is True, then the training
        coupon dataset is used. Otherwise, the test coupon dataset is used.
        """
        raise NotImplementedError



class ContentFilter(Model):

    def __init__(self, loader):
        """
        :param loader: DataLoader
        """
        super(ContentFilter, self).__init__(loader)


    def run(self, start, end):
        submission = []

        load = DataLoader()
        item_profile = ItemProfile(load)

        print "Userlist size: ", load.user_list.shape

        e = 0
        for index in load.user_list.index:
            user = User(load, index, item_profile)
            coupons = user.recommend()
            submission.append([user.get_id(), coupons])
            e += 1
            if e % 1000 == 0:
                print "At User: ", e

        return pd.DataFrame(submission, columns=["USER_ID_hash", "PURCHASED_COUPONS"])


    def predict(self, start, end, train=True):
        super(ContentFilter, self).predict(start, end, train)




































