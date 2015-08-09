from source_py.item import ItemProfile
from source_py.user import User

import pandas as pd


class Model(object):

    def __init__(self, train, test, users, purchases):
        """
        :param train: pandas.DataFrame of training coupon data
        :param test: pandas.DataFrame of test coupon data
        :param users: pandas.DataFrame of user data
        :param purchases: pandas.DataFrame of All user purchases
        """
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert isinstance(users, pd.DataFrame)
        assert isinstance(purchases, pd.DataFrame)

        self.users = users
        self.purchases = purchases

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
        self.categorical_weights = [2.0, 2.0,
                                    1.5, 1.5, 1.5]
        self._expand(self.categorical_weights)

        # scale numerical variables
        self.numerical = ["PRICE_RATE", "CATALOG_PRICE", "DISCOUNT_PRICE",
                       "VALIDPERIOD",
                       "USABLE_DATE_MON", "USABLE_DATE_TUE", "USABLE_DATE_WED",
                       "USABLE_DATE_THU", "USABLE_DATE_FRI", "USABLE_DATE_SAT",
                       "USABLE_DATE_SUN", "USABLE_DATE_HOLIDAY", "USABLE_DATE_BEFORE_HOLIDAY"]
        self.weights = pd.Series([2.0, 2.0, 2.0,
                                  1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0])
        self._scale(self.weights)

        # replace missing values
        self._replace_nan()

        # construct ItemProfile using finalized training and test sets
        self.item_profile = ItemProfile(self.train, self.test)


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
            user = User(self.users, index, self.purchases, self.train)
            coupons = self._recommend(user)
            submission.append([user.get_id(), coupons])
            e += 1
            if e % 1000 == 0:
                print "At User: ", e

        return pd.DataFrame(submission, columns=["USER_ID_hash", "PURCHASED_COUPONS"])


    def _recommend(self, user):
        """
        Recommends a ranked sequence of Coupons for a provided User.
        :param user: User
        """
        # get similarity scores for each test coupon
        scores = self.item_profile.similarity(user.coupons.index, self.test.index)
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
        coups = self.test.ix[top_indices].COUPON_ID_hash.tolist()
        # return as space-delimited string
        ids = ""
        for value in coups:
            ids += value + " "
        return ids.strip()


    def _expand(self, weights):
        """
        Expands categorical variables in training and test sets into
        0-1 dummy variables, and then scales them according to input weights.
        No future information is introduced from the test set.
        :param weights: list of scaling weights corresponding to categorical variables
        """
        # need to concatenate to maintain column consistency when expanding
        self.train["type"] = "train"
        self.test["type"] = "test"
        merged = self.train.append(self.test)

        # expand and scale categorical variables
        for field, weight in zip(self.categorical, weights):
            df = weight * pd.get_dummies(merged[field])
            merged = pd.concat([merged, df])
            merged.drop([field], inplace=True)

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



    def _scale(self, weights):
        """
        Normalizes all variables in the training and test sets using a set of weights.
        Note: NO future information is introduced. Test set normalization
        is done using training set min/max values.
        """
        df = self.train[self.numerical]
        train_min = df.min()
        train_max = df.max()
        self.train[self.numerical] = \
            (weights * ((df - train_min) / (train_max - train_min))) + 1.0
        df = self.test[self.numerical]
        self.test[self.numerical] = \
            (weights * ((df - train_min) / (train_max - train_min))) + 1.0


    def _replace_nan(self):
        """
        Replaces null/nan values.
        """
        self.train.fillna(value=0.5, inplace=True)
        self.test.fillna(value=0.5, inplace=True)


    def get_configuration(self):
        result = "Model Configuration:\n"
        result += "\nFields:\n" + str(self.fields)
        result += "\nVariable weights: " + str(self.weights)
        result += "\nCategorical:\n" + str(self.categorical)
        result += "\nItem Profile: Predict 10 items per user" + "\n"
        return result




































