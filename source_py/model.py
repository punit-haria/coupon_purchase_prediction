from source_py.item import ItemProfile
from source_py.user import User

import pandas as pd


class Model(object):

    def __init__(self, train, test, users, purchases, alpha=1.0):
        """
        :param train: pandas.DataFrame of training coupon data
        :param test: pandas.DataFrame of test coupon data
        :param users: pandas.DataFrame of user data
        :param purchases: pandas.DataFrame of All user purchases
        :param alpha: scaling factor for numerical variables
        """
        self.fields = None
        self.numerical = None
        self.categorical = None

        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert isinstance(users, pd.DataFrame)
        assert isinstance(purchases, pd.DataFrame)

        self.train = train
        self.test = test
        self.users = users
        self.purchases = purchases

        self.alpha = alpha


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


    def _expand(self):
        """
        Expands categorical variables in training and test sets into
        0-1 dummy variables. Note: NO future information is introduced
        from the test set.
        """
        # need to concatenate to maintain column consistency when expanding
        self.train["type"] = "train"
        self.test["type"] = "test"
        merged = self.train.append(self.test)

        # expand coupons
        merged = pd.get_dummies(merged, columns=self.categorical)
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


    def _scale(self):
        """
        Normalizes numerical variables in the training and test sets.
        Note: NO future information is introduced. Test set normalization
        is done using training set min/max values.
        """
        df = self.train[self.numerical]
        train_min = df.min()
        train_max = df.max()
        self.train[self.numerical] = self.alpha * (df - train_min) / (train_max - train_min)
        df = self.test[self.numerical]
        self.test[self.numerical] = self.alpha * (df - train_min) / (train_max - train_min)



class ContentFilter(Model):

    def __init__(self, train, test, users, purchases, alpha=1.0):
        """
        :param train: pandas.DataFrame of training coupon data
        :param test: pandas.DataFrame of test coupon data
        :param users: pandas.DataFrame of user data
        :param purchases: pandas.DataFrame of All user purchases
        """
        super(ContentFilter, self).__init__(train, test, users, purchases, alpha)

        self.fields = ["COUPON_ID_hash", "CAPSULE_TEXT", "GENRE_NAME", "PRICE_RATE",
                       "CATALOG_PRICE", "DISCOUNT_PRICE"]
        self.numerical = ["PRICE_RATE", "CATALOG_PRICE", "DISCOUNT_PRICE"]
        self.categorical = ["CAPSULE_TEXT", "GENRE_NAME"]

        # keep relevant coupon fields only
        self.train = self.train[self.fields].copy(deep=True)
        self.test = self.test[self.fields].copy(deep=True)

        self._scale()
        self._expand()

        # construct ItemProfile using finalized training and test sets
        self.item_profile = ItemProfile(self.train, self.test)


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


    def get_configuration(self):
        result = "Model Configuration:\n\n"
        result += "Fields:\n"
        for field in self.fields:
            result += field + "\t"
        result += "\nNumerical:\n"
        for field in self.numerical:
            result += field + "\t"
        result += "\nCategorical:\n"
        for field in self.categorical:
            result += field + "\t"
        result += "\nItem Profile: Predict 10 items per user"
        result += "\nNumerical Scaling factor: " + str(self.alpha)
        return result




































