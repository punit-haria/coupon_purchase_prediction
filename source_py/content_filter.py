import pandas as pd
from scipy.spatial.distance import cosine

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



class Coupons(object):

    def __init__(self, coupons):
        """
        :param coupons : pandas.DataFrame of coupons
        """
        self.coupons = coupons


    def similarity(self, input_coupons):
        """
        :param input_coupons : pandas.DataFrame of new coupons
        :return computes the cosine similarities of these coupons with the input coupons
        """
        test_coups = Coupons(input_coupons)

        test_df = test_coups.coupons.drop("COUPON_ID_hash", 1)
        df = self.coupons.drop("COUPON_ID_hash", 1)

        # get columns names
        cols = ["id"]
        for j, row in df.iterrows():
            cols.append(self.coupons.ix[j]["COUPON_ID_hash"])

        # get similarity scores
        results = []
        for i, test_row in test_df.iterrows():
            vals = [test_coups.coupons.ix[i]["COUPON_ID_hash"]]
            for j, row in df.iterrows():
                vals.append(1 - cosine(test_row, row))
            results.append(vals)

        return pd.DataFrame(results, columns=cols)



class User(object):

    def __init__(self, loader, index):
        """
        :param loader: DataLoader object with access to all data
        :param index: row index into user_list DataFrame for this user
        """
        self.data = loader
        self.users = self.data.user_list
        # this user
        self.user = self.users.ix[index]
        self.index = index
        # transactions
        self.purchases = self.data.details_train[self.data.details_train.USER_ID_hash == self.user.USER_ID_hash]
        # purchased coupons
        coupons_df = self.data.coupons_train[self.data.coupons_train.COUPON_ID_hash.isin(self.purchases.COUPON_ID_hash)]
        self.coupons = Coupons(coupons_df)


    def recommend(self):
        # get similarity scores for each test coupon
        scores = self.coupons.similarity(self.data.coupons_test)
        # compute mean similarity score for each test coupon
        scores["mean"] = scores.drop("id", axis=1).mean(axis=1)
        # 
        return scores


if __name__ == '__main__':

    load = DataLoader()
    user = User(load, 234)
    df = user.recommend()
























