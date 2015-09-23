__author__ = 'punit'

import pandas as pd


class LibfmLoader(object):

    def __init__(self, users, coupons_train, coupons_test, purchases):
        self.users = users
        self.coupons_train = coupons_train
        self.coupons_test = coupons_test
        self.purchases = purchases

    def convert(self):
        pass


def load():
    user_list = pd.read_csv("raw_data/user_list.csv")
    coupons_train = pd.read_csv("raw_data/coupon_list_train.csv")
    coupons_test = pd.read_csv("raw_data/coupon_list_test.csv")
    details_train = pd.read_csv("raw_data/coupon_detail_train.csv")

    return user_list, coupons_train, coupons_test, details_train


if __name__ == '__main__':

    print "hello"
