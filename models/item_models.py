from .models.item_matrix import ItemMatrix

import pandas as pd
import numpy as np

class Model(object):

    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    @staticmethod
    def _normalize(df):
        """
        Makes each column within range [0,1]
        :param df: DataFrame
        """
        new_df = pd.DataFrame()
        for col in df.columns:
            diff = df[col].max() - df[col].min()
            if diff == 0:
                new_df[col] = df[col].apply(lambda x : 0.5)
            else:
                new_df[col] = (df[col] - df[col].min()) / diff
        return new_df


class PurchaseModel(Model):

    def __init__(self, item_matrix, train_coupons, test_coupons, users, purchases):
        """
        :param item_matrix: object that contains item-item similarity matrix and other useful data
        :param train_coupons, test_coupons: training and test coupon data
        :param users: pandas.DataFrame of user data
        :param purchases: pandas.DataFrame of All user purchases
        """
        super(PurchaseModel, self).__init__()

        assert isinstance(item_matrix, ItemMatrix)
        self.im = item_matrix

        self.train = train_coupons
        self.test = test_coupons

        self.users = users
        self.purchases = purchases

        # parameters
        self.num_purchases_w = 0.15
        self.purchase_date_w = 0.85


    def run(self):
        """
        No training required.
        """
        pass


    def predict(self):
        """
        Returns scores of all test coupons for all users.
        """
        model_scores = []

        e = 0
        for index in self.users.index:
            # get user
            user = self.users.ix[index]
            # get relevant coupons and corresponding weights
            purchased_coupons, purchased_weights = self._coupon_filter(user)
            # get recommendations
            user_coupon_scores = self._recommend(purchased_coupons, purchased_weights)

            model_scores.append(user_coupon_scores)

            e += 1
            if e % 1000 == 0:
                print "At User: ", e

        return pd.DataFrame(model_scores, index=self.users.USER_ID_hash)


    def _recommend(self, purchased_coupons, purchased_weights):
        """
        Recommends a ranked sequence of Coupons for a provided User.
        :param purchased_coupons: user's purchased coupons
        :param purchased_weights: importance weights for each purchased coupon as a dictionary
        Returns all test coupons with their corresponding scores.
        """
        pscores = self.im.similarity(purchased_coupons.index, self.test.index)
        pscores = pscores.transpose() # now: (test coupons, user coupons)
        if pscores.shape[1] == 0:
            pscores["mean"] = 0
        else:
            # compute similarity score for each test coupon using purchased coupons and weights
            pscores["mean"] = pscores.dot(purchased_weights)

        return pscores["mean"].transpose().to_dict()


    def _coupon_filter(self, user):
        """
        :param user: row corresponding to user in user_list
        Takes the user and returns the purchased coupons and visited coupons along
        with their corresponding weights.
        """
        # get user purchases (note: not all users have made purchases)
        user_buys = self.purchases[self.purchases.USER_ID_hash == user.USER_ID_hash]
        # get corresponding purchased coupons
        purchased_coupons = self.train[self.train.COUPON_ID_hash.isin(user_buys.COUPON_ID_hash)]

        final_purchased_weights = None
        if not purchased_coupons.empty:
            # get the frequency of purchase for each coupon
            bought_coupon_groups = user_buys.groupby(by='COUPON_ID_hash').groups
            purchased_weights = {}
            for key in bought_coupon_groups:
                new_key = purchased_coupons[purchased_coupons.COUPON_ID_hash == key].index[0]
                purchased_weights[new_key] = len(bought_coupon_groups[key])
            purchased_weights = pd.DataFrame.from_dict(purchased_weights, orient='index').sort_index()

            purchased_weights.columns = ["freq"]
            purchased_weights = Model._normalize(purchased_weights) # make range [0,1]

            # get the most recent purchase date for each coupon
            pdates = user_buys[["COUPON_ID_hash", "NUM_DAYS"]].groupby(by='COUPON_ID_hash').max()
            pdates.columns = ["recent"]
            pdates = Model._normalize(pdates) # make range [0,1]
            actual_index = []
            for coup in pdates.index:
                actual_index.append(purchased_coupons[purchased_coupons.COUPON_ID_hash == coup].index[0])
            pdates.index = np.array(actual_index)
            pdates.sort_index(inplace=True)

            # sanity check
            assert pdates.shape == purchased_weights.shape
            assert pdates.index.equals(purchased_weights.index)

            # generate final purchase weights
            final_purchased_weights = (self.num_purchases_w * purchased_weights["freq"]) + \
                                  (self.purchase_date_w * pdates["recent"])

        return purchased_coupons, final_purchased_weights
