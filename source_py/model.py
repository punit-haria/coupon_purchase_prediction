from source_py.data import DataLoader
from source_py.item import ItemProfile
from source_py.user import User

import pandas as pd


class Model(object):

    def __init__(self):
        pass

    def run(self):
        """
        Run model training process.
        """
        raise NotImplementedError

    def predict(self):
        """
        Predictions are made for test set.
        """
        raise NotImplementedError



class ContentFilter(Model):

    def __init__(self, users, train, test, purchases):
        """
        :param users: pandas.DataFrame of user data
        :param train: pandas.DataFrame of training coupon data
        :param test: pandas.DataFrame of test coupon data
        :param purchases: pandas.DataFrame of All user purchases
        """
        super(ContentFilter, self).__init__()
        assert isinstance(users, pd.DataFrame)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert isinstance(purchases, pd.DataFrame)

        self.train = train
        self.test = test
        self.users = users
        self.purchases = purchases

        self.item_profile = ItemProfile(train, test)


    def run(self):
        """
        Run model training process.
        """
        print "No training required..."


    def predict(self):
        """
        Predictions are made on the test set; returns DataFrame in Kaggle submission format.
        """
        submission = []

        print "Userlist size: ", self.users.shape

        e = 0
        for index in self.users.index:
            user = User(self.users, index, self.purchases, self.train)
            coupons = self._recommend(user, self.test)
            submission.append([user.get_id(), coupons])
            e += 1
            if e % 1000 == 0:
                print "At User: ", e

        return pd.DataFrame(submission, columns=["USER_ID_hash", "PURCHASED_COUPONS"])


    def _recommend(self, user, test_coupons):
        """
        Recommends a ranked sequence of Coupons for a provided User.
        :param user: User
        :param test_coupons: pandas.DataFrame of test coupon data
        """
        # get similarity scores for each test coupon
        scores = self.item_profile.similarity(user.coupons.index, test_coupons.index)
        # x-axis: test coupons, y-axis: user coupons
        scores = scores.transpose()
        # compute mean similarity score for each test coupon
        scores["mean"] = scores.mean(axis=1)
        # sort by descending order of mean score
        scores.sort(columns="mean", ascending=False, inplace=True)
        # get top test coupon indices
        top = User.num_coupons()
        if top < 1:
            return ""
        top_indices = scores.head(n=top).index
        # get top coupon IDs
        coups = test_coupons.ix[top_indices].COUPON_ID_hash.tolist()
        # return as space-delimited string
        ids = ""
        for value in coups:
            ids += value + " "
        return ids




































