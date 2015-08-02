import os.path
import numpy as np
import pandas as pd
from numpy.linalg import norm


class DataLoader(object):

    fields = ["COUPON_ID_hash", "CAPSULE_TEXT", "GENRE_NAME", "PRICE_RATE",
                       "CATALOG_PRICE", "DISCOUNT_PRICE"]
    numerical = ["PRICE_RATE", "CATALOG_PRICE", "DISCOUNT_PRICE"]
    categorical = ["CAPSULE_TEXT", "GENRE_NAME"]

    def __init__(self, alpha=1.0):
        """
        :param alpha: scaling factor for numerical variables in Item Profile
        """
        self.user_list = pd.read_csv("raw_data/user_list.csv")
        self.coupons_train = pd.read_csv("raw_data/coupon_list_train.csv")
        self.coupons_test = pd.read_csv("raw_data/coupon_list_test.csv")
        self.details_train = pd.read_csv("raw_data/coupon_detail_train.csv")

        # ensure data is read correctly
        assert self.user_list.shape == (22873,6)
        assert self.coupons_train.shape == (19413,24)
        assert self.coupons_test.shape == (310,24)
        assert self.details_train.shape == (168996,6)

        # keep relevant coupon fields only
        self.coupons_train = self.coupons_train[DataLoader.fields]
        self.coupons_test = self.coupons_test[DataLoader.fields]

        # scale coupons
        self.alpha = alpha
        self.scale()

        # need to concatenate to maintain column consistency when expanding
        self.coupons_train["type"] = "train"
        self.coupons_test["type"] = "test"
        merged = self.coupons_train.append(self.coupons_test)

        # expand coupons
        merged = self.expand(merged)
        self.coupons_train = pd.DataFrame.copy(merged[merged.type == "train"], deep=True)
        self.coupons_train.reset_index(inplace=True)
        self.coupons_train.drop(["index","type"], axis=1, inplace=True)
        self.coupons_test = pd.DataFrame.copy(merged[merged.type == "test"], deep=True)
        self.coupons_test.reset_index(inplace=True)
        self.coupons_test.drop(["index","type"], axis=1, inplace=True)

        # validate coupon expansion
        assert len(self.coupons_train.columns) == len(self.coupons_test.columns)
        for left, right in zip(self.coupons_train.columns, self.coupons_test.columns):
            assert left == right


    def scale(self):
        """
        Normalizes numerical variables in the training and test sets.
        """
        def normalize(df, scale):
            return scale * (df - df.min()) / (df.max() - df.min())

        self.coupons_train[DataLoader.numerical] = normalize(self.coupons_train[DataLoader.numerical], self.alpha)
        self.coupons_test[DataLoader.numerical] = normalize(self.coupons_test[DataLoader.numerical], self.alpha)


    @staticmethod
    def expand(df):
        """
        Expands the categorical variables of input DataFrame into dummy variables.
        """
        return pd.get_dummies(df, columns=DataLoader.categorical)



class ItemProfile(object):

    def __init__(self, loader, simfn="cosine"):
        """
        :param loader: DataLoader
        :param simfn : similarity function
        """
        self.data = loader
        self.path = "data/item_similarities.csv"
        self.matrix = None
        if simfn == "cosine":
            self.simfn = self._generate_cosine
        else:
            raise NotImplementedError


    def similarity(self, train_range, test_range):
        """
        :param train_range, test_range : lists of indices into training and test coupon DataFrames
        :return the cosine similarities of these coupons with the input coupons
        """
        simil_matrix = self.generate()
        return simil_matrix.ix[train_range][test_range]


    def generate(self):
        """
        Returns the Item-Item similarity matrix between training and test coupons.
        """
        if self.matrix is not None:
            return self.matrix
        elif os.path.isfile(self.path):
            self.matrix = pd.read_csv(self.path)
        else:
            self.matrix = self.simfn()

        return self.matrix


    def _generate_cosine(self):
        """
        Generates an Item-Item cosine similarity matrix between the training and test coupons.
        """
        train = self.data.coupons_train.drop("COUPON_ID_hash", 1)
        test = self.data.coupons_test.drop("COUPON_ID_hash", 1)

        # normalize data
        train = train.div(train.apply(norm, axis=1), axis='index')
        test = test.div(test.apply(norm, axis=1), axis='index')

        # compute cosine similarities
        return train.dot(test.transpose())


class User(object):

    def __init__(self, loader, index, item_profile):
        """
        :param loader: DataLoader object with access to all data
        :param index: row index into user_list DataFrame for this user
        :param item_profile : ItemProfile object
        """
        self.data = loader
        # this user
        self.user = self.data.user_list.ix[index]
        self.index = index
        # transactions
        self.purchases = self.data.details_train[self.data.details_train.USER_ID_hash == self.user.USER_ID_hash]
        # purchased coupons
        self.coupons = self.data.coupons_train[self.data.coupons_train.COUPON_ID_hash.isin(self.purchases.COUPON_ID_hash)]
        # item profile object
        self.item_profile = item_profile

    def get_id(self):
        """
        :return: ID of this user
        """
        return self.user.USER_ID_hash


    def recommend(self):
        """
        :return: gets the recommended coupons from the test set for this user
        """
        # get similarity scores for each test coupon
        scores = self.item_profile.similarity(self.coupons.index, self.data.coupons_test.index)
        # x-axis: test coupons, y-axis: user coupons
        scores = scores.transpose()
        # compute mean similarity score for each test coupon
        scores["mean"] = scores.mean(axis=1)
        # sort by descending order of mean score
        scores.sort(columns="mean", ascending=False, inplace=True)
        # get top test coupon indices
        top_indices = scores.head(n=User.num_coupons()).index
        # get top coupon IDs
        coups = self.data.coupons_test.ix[top_indices].COUPON_ID_hash.tolist()
        # return as space-delimited string
        ids = ""
        for value in coups:
            ids += value + " "
        return ids


    @staticmethod
    def num_coupons():
        """
        :return: the number of coupons to recommend for this user
        """
        return 5


def run(output_filename):
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
    final_df = pd.DataFrame(submission, columns=["USER_ID_hash", "PURCHASED_COUPONS"])

    final_df.to_csv(output_filename, sep=",", index=False, header=True)



if __name__ == '__main__':

    run("optimized_output.csv")


























