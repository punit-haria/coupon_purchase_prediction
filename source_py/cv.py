from source_py.data import DataLoader
from source_py.model import ContentFilter

import pandas as pd
import numpy as np

class Validator(object):

    def __init__(self, start_date, training_period, validation_period, loader):
        """
        :param start_date: string for start date for training period (format "%Y-%m-%d")
        :param training_period: length of training period in days
        :param validation_period: length of validation period in days
        :param loader: DataLoader
        """
        print "Creating training and test sets..."

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

        print "Splitting user transactions..."

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

        print "Getting actual purchases..."

        # actual purchases/recommendations in Kaggle submission format
        self.actual = self._actual_purchases()

        self.model = None


    def run(self, mode="content_filter"):
        """
        :param mode: Model subclass
        Trains the model on the training set and returns predictions for the test set.
        """
        if mode == "content_filter":
            print "Initializing model..."
            self.model = ContentFilter(self.train, self.test, self.users, self.purchases)
            print "Training model..."
            self.model.run()
            print "Making predictions..."
            return self.model.predict()
        else:
            raise NotImplementedError


    def _actual_purchases(self):
        """
        Returns a pandas.DataFrame of the actual coupon purchases during the validation period.
        Format of the DataFrame is identical to the kaggle submission format.
        """
        results = []
        tp = self.test_purchases
        for index in self.users.index:
            user_id = self.users.ix[index].USER_ID_hash
            user_purchases = tp[tp.USER_ID_hash == user_id].sort("ITEM_COUNT", ascending=False, axis=0)
            coups = user_purchases.COUPON_ID_hash.tolist()
            ids = ""
            for value in coups:
                ids += value + " "
            ids = ids.strip()
            results.append([user_id, ids])
        return pd.DataFrame(results, columns=["USER_ID_hash", "PURCHASED_COUPONS"])


    def mapk(self, k, actual, predicted):
        """
        :param k: max length of predicted sequence
        :param actual: DataFrame of actual purchases for each user (kaggle format)
        :param predicted: DataFrame of predicted purchases for each user (kaggle format)
        :return: Mean Average Precision at k

        See https://github.com/benhamner/Metrics/blob/master/R/R/metrics.r
        """
        print "Computing MAP score..."
        scores = []
        for i, j in zip(actual.index, predicted.index):
            a = actual.ix[i].PURCHASED_COUPONS
            p = predicted.ix[j].PURCHASED_COUPONS
            scores.append(self.apk(k, a, p))
        return np.array(scores).mean()


    @staticmethod
    def apk(k, actual, predicted):
        """
        :param k: max length of predicted sequence
        :param actual: actual Coupon hash tags as a list
        :param predicted: list of predicted Coupon hash tags
        :return: Average Precision at k
        """
        actual = actual.split(' ')
        predicted = predicted.split(' ')
        score = 0.0
        cnt = 0.0
        for i in range(min(k, len(predicted))):
            if predicted[i] in actual:
                if predicted[i] not in predicted[0:i]:
                    cnt += 1
                    score += cnt/(i+1)
        return score / min(len(actual),k)





