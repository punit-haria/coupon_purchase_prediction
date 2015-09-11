from source_py.item import ItemProfile
from source_py.timer import Timer

import pandas as pd
import numpy as np


class PurchasedCouponModel(object):

    def __init__(self, train, test, users, purchases):
        """
        :param train: pandas.DataFrame of training coupon data
        :param test: pandas.DataFrame of test coupon data
        :param users: pandas.DataFrame of user data
        :param purchases: pandas.DataFrame of All user purchases
        :param visits: pandas.DataFrame of All user visits
        """
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
        self.train = train[self.fields]
        self.test = test[self.fields]

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

