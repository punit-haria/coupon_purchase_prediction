__author__ = 'punit'

import pandas as pd
import math
import sys

# load all data
def load():
    user_list = pd.read_csv("raw_data/user_list.csv")
    coupons_train = pd.read_csv("raw_data/coupon_list_train.csv")
    coupons_test = pd.read_csv("raw_data/coupon_list_test.csv")
    details_train = pd.read_csv("raw_data/coupon_detail_train.csv")

    return user_list, coupons_train, coupons_test, details_train

class LibfmLoader(object):

    def __init__(self, users, coupons_train, coupons_test, purchases):
        self.users = users
        self.coupons_train = coupons_train
        self.coupons_test = coupons_test
        self.purchases = purchases

        self.coupons_train["type"] = "train"
        self.coupons_test["type"] = "test"
        self.coupons = self.coupons_train.append(self.coupons_test)
        self.coupons.reset_index(inplace=True)

        self.num_users = self.users.shape[0]
        self.num_train_items = self.coupons_train.shape[0]
        self.num_test_items = self.coupons_test.shape[0]
        self.num_items = self.num_train_items + self.num_test_items

        self.result = pd.DataFrame()


    def convert_train(self):

        print "adding target..."

        self.result["target"] = 1.0

        print "converting user, item indicators..."

        def user_indicator(df):
            return str(self.users[self.users.USER_ID_hash == df.USER_ID_hash].index[0])+":1"

        def coupon_indicator(df):
            real_index = self.coupons[self.coupons.COUPON_ID_hash == df.COUPON_ID_hash].index[0] + self.num_users
            return str(real_index)+":1"

        self.result["user"]  = self.purchases.apply(user_indicator, axis=1)
        self.result["item"] = self.purchases.apply(coupon_indicator, axis=1)

        print "converting purchased item indicators..."

        def similar_items_indicator(df):
            uid_hash = self.users.ix[int(str(df.user).split(':')[0])]['USER_ID_hash']
            purchased_set = self.purchases[self.purchases.USER_ID_hash == uid_hash]["COUPON_ID_hash"].unique().tolist()
            fstr = ""
            value = str(1.0 / math.sqrt(len(purchased_set)))
            for item in purchased_set:
                fstr += str(self.coupons[self.coupons.COUPON_ID_hash == item].index[0] + self.num_users + self.num_items)
                fstr += ":" + value + " "
            return fstr

        self.result["other_items"] = self.result.apply(similar_items_indicator, axis=1)

        print "adding negative target values..."
        neg_values = pd.DataFrame()
        neg_values["target"] = 0.0

        user_list = self.users.USER_ID_hash.tolist()
        item_list = self.coupons_train.COUPON_ID_hash.tolist()

        print "adding negative user,item indicators..."
        print "Number of users: ", self.num_users
        print "Number of items: ", self.num_train_items
        e = 0
        user_indicator_list = []
        item_indicator_list = []
        simil_indicator_list = []
        for user in user_list:
            uval = str(self.users[self.users.USER_ID_hash == user].index[0]) + ":1"
            for item in item_list:
                if self.purchases[(self.purchases.USER_ID_hash == user) & (self.purchases.COUPON_ID_hash == item)].empty:
                    # get user value
                    user_indicator_list.append(uval)
                    # get item value
                    ival = str(self.coupons[self.coupons.COUPON_ID_hash == item].index[0] + self.num_users) + ":1"
                    item_indicator_list.append(ival)
                    # get similar items
                    sval = str(self.result[self.result.users == uval]["other_items"].unique()[0])
                    simil_indicator_list.append(sval)

            e += 1
            if e % 1000 == 0:
                print "At user: ", e

        neg_values["user"] = pd.Series(user_indicator_list)
        neg_values["item"] = pd.Series(item_indicator_list)
        neg_values["other_items"] = pd.Series(simil_indicator_list)

        self.result.append(neg_values)


    def write(self, train_output_fp, test_output_fp):
        print "writing to file..."
        self.result.to_csv(train_output_fp, sep=" ", header=False, index=False)


if __name__ == '__main__':

    train_output_path = sys.argv[1]
    test_output_path = sys.argv[2]

    users, coupons_train, coupons_test, purchases = load()

    libfm = LibfmLoader(users, coupons_train, coupons_test, purchases.ix[0:10])
    libfm.convert_train()
    libfm.write(train_output_path, test_output_path)


