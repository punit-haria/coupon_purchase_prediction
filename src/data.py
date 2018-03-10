import pandas as pd
import numpy as np

# Load Data 


class DataLoader(object):

    def __init__(self):

        self.user_list = pd.read_csv("raw_data/user_list.csv")
        self.coupons_train = pd.read_csv("raw_data/coupon_list_train.csv")
        self.coupons_test = pd.read_csv("raw_data/coupon_list_test.csv")
        self.details_train = pd.read_csv("raw_data/coupon_detail_train.csv")

        self.listing_area_train = pd.read_csv("raw_data/coupon_area_train.csv")
        self.listing_area_test = pd.read_csv("raw_data/coupon_area_test.csv")

        self.locations = pd.read_csv("raw_data/prefecture_locations.csv")

        self.visits = pd.read_csv("raw_data/coupon_visit_train.csv")

        # ensure data is read correctly
        assert self.user_list.shape == (22873,6)
        assert self.coupons_train.shape == (19413,24)
        assert self.coupons_test.shape == (310,24)
        assert self.details_train.shape == (168996,6)

        assert self.listing_area_train.shape == (138185,3)
        assert self.listing_area_test.shape == (2165,3)

        assert self.locations.shape == (47, 4)

        assert self.visits.shape == (2833180, 8)

        # drop small_area_name from area listings because we can't map it to coordinates
        self.listing_area_train.drop("SMALL_AREA_NAME", axis=1, inplace=True)
        self.listing_area_test.drop("SMALL_AREA_NAME", axis=1, inplace=True)
        # remove listing duplicates (since column is dropped)
        self.listing_area_train.drop_duplicates(inplace=True)
        self.listing_area_test.drop_duplicates(inplace=True)

        # join coordinate data to coupon area listings data
        self.locations_train = self.listing_area_train.merge(self.locations,
                                                             how='left', on='PREF_NAME')
        self.locations_test = self.listing_area_test.merge(self.locations,
                                                             how='left', on='PREF_NAME')

        # convert variables to datetime format
        fmt = '%Y-%m-%d %H:%M:%S'
        self.coupons_train.DISPFROM = pd.to_datetime(self.coupons_train.DISPFROM, format=fmt)
        self.details_train.I_DATE = pd.to_datetime(self.details_train.I_DATE, format=fmt)
        self.visits.I_DATE = pd.to_datetime(self.visits.I_DATE, format=fmt)

        # number of days since beginning
        self.details_train["NUM_DAYS"] \
            = ((self.details_train.I_DATE - self.details_train.I_DATE.min()) / np.timedelta64(1, 'D')).astype(int)
        self.visits["NUM_DAYS"] \
            = ((self.visits.I_DATE - self.visits.I_DATE.min()) / np.timedelta64(1, 'D')).astype(int)


