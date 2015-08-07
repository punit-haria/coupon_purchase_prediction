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

        # convert variables to datetime format
        fmt = '%Y-%m-%d %H:%M:%S'
        self.coupons_train.DISPFROM = pd.to_datetime(self.coupons_train.DISPFROM, format=fmt)
        self.details_train.I_DATE = pd.to_datetime(self.details_train.I_DATE, format=fmt)


