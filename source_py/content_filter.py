import pandas as pd

class DataLoader(object):
    def __init__(self):
        self.user_list = pd.read_csv("raw_data/user_list.csv")
        self.coupons_train = pd.read_csv("raw_data/coupon_list_train.csv")
        self.coupons_test = pd.read_csv("raw_data/coupon_list_test.csv")
        self.details_train = pd.read_csv("raw_data/coupon_detail_train.csv")

        # ensure data is read correctly
        assert self.user_list.shape == (22873,6)
        assert self.coupons_train.shape == (19413,24)
        assert self.coupons_test.shape == (310,24)
        assert self.details_train.shape == (168996,6)


class ItemProfile(object):

    def __init__(self, coupons_train, coupons_test):
        """
        :param coupons_train : pandas.DataFrame of coupons for training
        :param coupons_test : pandas.DataFrame of coupons for testing
        """
        self.raw_train = coupons_train
        self.raw_test = coupons_test
        self.fields = ["CAPSULE_TEXT", "GENRE_NAME", "PRICE_RATE",
                       "CATALOG_PRICE", "DISCOUNT_PRICE"]

        self.train = pd.DataFrame.copy(coupons_train[self.fields], deep=True)
        self.test = pd.DataFrame.copy(coupons_test[self.fields], deep=True)

    def scale(self, alpha=1.0):
        """
        Normalizes numerical variables in the training and test sets.
        :param alpha : scaling factor
        """
        def normalize(df, scale):
            return scale * (df - df.min()) / (df.max() - df.min())

        numerical = ["PRICE_RATE", "CATALOG_PRICE", "DISCOUNT_PRICE"]
        self.train[numerical] = normalize(self.train[numerical], alpha)
        self.test[numerical] = normalize(self.test[numerical], alpha)

    def expand(self):
        """
        Expands the categorical variables of the training and test sets into
        dummy variables.
        """
        self.train = pd.get_dummies(self.train)
        self.test = pd.get_dummies(self.test)

    @staticmethod
    def similarity(self, left, right):
        """
        :param left : left vector
        :param right : right vector
        :return computes the cosine similarity of the two input Item vectors
        """
        raise NotImplementedError



if __name__ == '__main__':

    print "hello"























