import pandas as pd


class DataLoader(object):

    fields = ["COUPON_ID_hash", "CAPSULE_TEXT", "GENRE_NAME", "PRICE_RATE",
                       "CATALOG_PRICE", "DISCOUNT_PRICE"]
    numerical = ["PRICE_RATE", "CATALOG_PRICE", "DISCOUNT_PRICE"]
    categorical = ["CAPSULE_TEXT", "GENRE_NAME"]


    def __init__(self, alpha=1.0):
        """
        :param alpha: scaling factor for numerical variables
        """
        self.alpha = alpha

        # read raw data
        self.__read_raw_data()

        # keep relevant coupon fields only
        self.coupons_train = self.coupons_train[DataLoader.fields]
        self.coupons_test = self.coupons_test[DataLoader.fields]

        # scale numerical variables
        self._scale()

        # expand categorical variables
        self._expand_data()


    def __read_raw_data(self):
        self.user_list = pd.read_csv("raw_data/user_list.csv")
        self.coupons_train = pd.read_csv("raw_data/coupon_list_train.csv")
        self.coupons_test = pd.read_csv("raw_data/coupon_list_test.csv")
        self.details_train = pd.read_csv("raw_data/coupon_detail_train.csv")

        # ensure data is read correctly
        assert self.user_list.shape == (22873,6)
        assert self.coupons_train.shape == (19413,24)
        assert self.coupons_test.shape == (310,24)
        assert self.details_train.shape == (168996,6)


    def _expand_data(self):
        def expand(df):
            """
            Expands the categorical variables of input DataFrame into dummy variables.
            """
            return pd.get_dummies(df, columns=DataLoader.categorical)

        # need to concatenate to maintain column consistency when expanding
        self.coupons_train["type"] = "train"
        self.coupons_test["type"] = "test"
        merged = self.coupons_train.append(self.coupons_test)

        # expand coupons
        merged = expand(merged)
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


    def _scale(self):
        """
        Normalizes numerical variables in the training and test sets.
        """
        def normalize(df, scale):
            return scale * (df - df.min()) / (df.max() - df.min())

        self.coupons_train[DataLoader.numerical] = normalize(self.coupons_train[DataLoader.numerical], self.alpha)
        self.coupons_test[DataLoader.numerical] = normalize(self.coupons_test[DataLoader.numerical], self.alpha)




