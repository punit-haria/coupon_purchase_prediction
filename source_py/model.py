from source_py.item import ItemProfile
from source_py.timer import Timer

import pandas as pd
import numpy as np

class Model(object):

    def __init__(self, train, test, users, purchases, visits):
        """
        :param train: pandas.DataFrame of training coupon data
        :param test: pandas.DataFrame of test coupon data
        :param users: pandas.DataFrame of user data
        :param purchases: pandas.DataFrame of All user purchases
        :param visits: pandas.DataFrame of All user visits
        """
        self.users = users
        self.purchases = purchases

        self.visits = visits[visits["PURCHASE_FLG"] == 1].copy(deep=True)
        self.visits = self.visits[self.visits.VIEW_COUPON_ID_hash.isin(train.COUPON_ID_hash)]

        self.fields = ["COUPON_ID_hash",
                       "CAPSULE_TEXT", "GENRE_NAME",
                       "PRICE_RATE", "CATALOG_PRICE", "DISCOUNT_PRICE",
                       "VALIDPERIOD",
                       "USABLE_DATE_MON", "USABLE_DATE_TUE", "USABLE_DATE_WED",
                       "USABLE_DATE_THU", "USABLE_DATE_FRI", "USABLE_DATE_SAT",
                       "USABLE_DATE_SUN", "USABLE_DATE_HOLIDAY", "USABLE_DATE_BEFORE_HOLIDAY",
                       "large_area_name", "ken_name", "small_area_name"]

        # keep relevant coupon fields only
        self.train = train[self.fields].copy(deep=True)
        self.test = test[self.fields].copy(deep=True)

        # expand categorical variables
        self.categorical = ["CAPSULE_TEXT", "GENRE_NAME",
                            "large_area_name", "ken_name", "small_area_name"]
        self.categorical_weights = [3.0, 3.0,
                                    3.0, 3.0, 3.0]
        self._expand(self.categorical_weights, transform=1.0)

        # scale numerical variables
        self.numerical = ["PRICE_RATE", "CATALOG_PRICE", "DISCOUNT_PRICE",
                       "VALIDPERIOD",
                       "USABLE_DATE_MON", "USABLE_DATE_TUE", "USABLE_DATE_WED",
                       "USABLE_DATE_THU", "USABLE_DATE_FRI", "USABLE_DATE_SAT",
                       "USABLE_DATE_SUN", "USABLE_DATE_HOLIDAY", "USABLE_DATE_BEFORE_HOLIDAY"]
        self.numerical_weights = [1.0, 1.0, 1.0,
                        1.0,
                        1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0]
        self._scale(self.numerical_weights, transform=1.0)

        # replace missing values
        self._replace_nan(0.)

        self.timer = Timer()

        # construct ItemProfile using finalized training and test sets
        self.item_profile = ItemProfile(self.train, self.test)

        # parameters
        self.num_purchases_w = 0.15
        self.purchase_date_w = 0.85
        self.purchased_w = 0.7
        self.visited_w = 0.3



    @staticmethod
    def run():
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
        print "Training coupons: ", self.train.shape
        print "Test coupons: ", self.test.shape

        e = 0
        for index in self.users.index:
            # get user
            user = self.users.ix[index]
            # get relevant coupons and corresponding weights
            purchased_coupons, purchased_weights, visited_coupons, visited_weights = self._coupon_filter(user)
            # get recommendations
            final_coupons = self._recommend(purchased_coupons, purchased_weights, visited_coupons, visited_weights)
            # add to submissions
            submission.append([user.USER_ID_hash, final_coupons])
            e += 1
            if e % 1000 == 0:
                print "At User: ", e

        return pd.DataFrame(submission, columns=["USER_ID_hash", "PURCHASED_COUPONS"])


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

        # get user visits (not including purchases)
        user_visits = self.visits[self.visits.USER_ID_hash == user.USER_ID_hash]
        # get corresponding visited coupons
        visited_coupons = self.train[self.train.COUPON_ID_hash.isin(user_visits.VIEW_COUPON_ID_hash)]

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
            purchased_weights = Model._normalize(purchased_weights) # scale to [0,1]

            # get the most recent purchase date for each coupon
            pdates = user_buys[["COUPON_ID_hash", "NUM_DAYS"]].groupby(by='COUPON_ID_hash').max()
            pdates.columns = ["recent"]
            pdates = Model._normalize(pdates) # scale to [0,1]
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

        final_visited_weights = None
        if not visited_coupons.empty:
            # get the most recent visitation date for each coupon
            vdates = user_visits[["VIEW_COUPON_ID_hash", "NUM_DAYS"]].groupby(by='VIEW_COUPON_ID_hash').max()
            vdates.columns = ["recent"]
            vdates = Model._normalize(vdates) # scale to [0,1]
            actual_vindex = []
            for coup in vdates.index:
                actual_vindex.append(visited_coupons[visited_coupons.COUPON_ID_hash == coup].index[0])
            vdates.index = np.array(actual_vindex)
            vdates.sort_index(inplace=True)
            final_visited_weights = vdates.recent

        return purchased_coupons, final_purchased_weights, visited_coupons, final_visited_weights


    def _recommend(self, purchased_coupons, purchased_weights, visited_coupons, visited_weights):
        """
        Recommends a ranked sequence of Coupons for a provided User.
        :param purchased_coupons: user's purchased coupons
        :param purchased_weights: importance weights for each purchased coupon as a dictionary
        """
        pscores = self.item_profile.similarity(purchased_coupons.index, self.test.index)
        pscores = pscores.transpose() # now: (test coupons, user coupons)
        if pscores.shape[1] == 0:
            pscores["pmean"] = 0
        else:
            # compute similarity score for each test coupon using purchased coupons and weights
            pscores["pmean"] = pscores.dot(purchased_weights)

        # get similarity scores for each test coupon
        vscores = self.item_profile.similarity(visited_coupons.index, self.test.index)
        vscores = vscores.transpose() # now: (test coupons, user coupons)
        if vscores.shape[1] == 0:
            vscores["vmean"] = 0
        else:
            # compute similarity score for each test coupon using purchased coupons and weights
            vscores["vmean"] = vscores.dot(visited_weights)

        # combine scores
        scores = pd.concat([pscores, vscores], axis=1)
        scores["mean"] = (self.purchased_w * pscores["pmean"]) + (self.visited_w * vscores["vmean"])
        # sort by descending order of mean score
        scores.sort(columns="mean", ascending=False, inplace=True)
        # get top test coupon indices
        top_indices = scores.head(n=10).index
        # get top coupon IDs
        coups = self.test.ix[top_indices].COUPON_ID_hash.tolist()

        # return as space-delimited string
        ids = ""
        for value in coups:
            ids += value + " "
        return ids.strip()


    def _expand(self, weights, transform):
        """
        Expands categorical variables in training and test sets into
        0-1 dummy variables, scales them according to input weights, and
        finally transforms upwards slightly.
        No future information is introduced from the test set.
        :param weights: list of scaling weights corresponding to categorical variables
        """
        # need to concatenate to maintain column consistency when expanding
        self.train["type"] = "train"
        self.test["type"] = "test"
        merged = self.train.append(self.test)

        # expand and scale categorical variables
        for field, weight in zip(self.categorical, weights):
            df = (weight * pd.get_dummies(merged[field])) + transform
            merged = pd.concat([merged, df], axis=1)

        # drop original categoricals
        merged.drop(self.categorical, axis=1, inplace=True)

        # split back into training and test sets
        self.train = pd.DataFrame.copy(merged[merged.type == "train"], deep=True)
        self.train.reset_index(inplace=True)
        self.train.drop(["index","type"], axis=1, inplace=True)

        self.test = pd.DataFrame.copy(merged[merged.type == "test"], deep=True)
        self.test.reset_index(inplace=True)
        self.test.drop(["index","type"], axis=1, inplace=True)

        # validate coupon expansion
        assert len(self.train.columns) == len(self.test.columns)
        for left, right in zip(self.train.columns, self.test.columns):
            assert left == right



    def _scale(self, weights, transform):
        """
        Normalizes all variables in the training and test sets using a set of weights.
        Variables are then transformed upwards slightly.
        Note: NO future information is introduced. Test set normalization
        is done using training set min/max values.
        """
        # scale between 0 and 1
        df = self.train[self.numerical]
        train_min = df.min()
        train_max = df.max()
        self.train[self.numerical] = (df - train_min) / (train_max - train_min)
        df = self.test[self.numerical]
        self.test[self.numerical] = (df - train_min) / (train_max - train_min)

        # apply weights and transform
        for field, weight in zip(self.numerical, weights):
            self.train[field] = (weight * self.train[field]) + transform
            self.test[field] = (weight * self.test[field]) + transform


    def _replace_nan(self, value):
        """
        Replaces null/nan values.
        """
        self.train.fillna(value=value, inplace=True)
        self.test.fillna(value=value, inplace=True)


    @staticmethod
    def _normalize(df):
        """
        Makes column of resulting DataFrame have min:0 and max:1
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


    def get_configuration(self):
        result = "Model Configuration:\n"
        result += "\nFields:\n" + str(self.fields)
        result += "\n\nCategorical:\n" + str(self.categorical)
        result += "\nWeights: " + str(self.categorical_weights)
        result += "\n\nNumerical:\n" + str(self.numerical)
        result += "\nWeights: " + str(self.numerical_weights)
        result += "\nItem Profile: Predict 10 items per user"
        result += "\nTrain data: "+str(self.train.shape)
        result += "\nTest data: "+str(self.test.shape)
        result += "\nUser list: "+str(self.users.shape)
        result += "\nPurchases: "+str(self.purchases.shape)
        result += "\n\nParameters:"
        result += "\nnum_purchases_w: "+str(self.num_purchases_w)
        result += "\npurchase_date_w: "+str(self.purchase_date_w)
        result += "\npurchase_w: "+str(self.purchased_w)
        result += "\nvisited_w: "+str(self.visited_w)

        return result




































