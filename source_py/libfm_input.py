__author__ = 'punit'

import pandas as pd
import os.path
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

    def __init__(self, users, coupons_train, coupons_test, purchases, reset_index=False):
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
        self.result_test = pd.DataFrame()

        print "getting user,item index..."
        self.uindex_fname = "datalibfm/user_dict.txt"
        self.iindex_fname = "datalibfm/item_dict.txt"
        if reset_index:
            self.uindex, self.iindex = self._generate_index()
        elif os.path.isfile(self.uindex_fname) and os.path.isfile(self.uindex_fname):
            self.uindex = pd.read_csv(self.uindex_fname, header=None)
            self.iindex = pd.read_csv(self.iindex_fname, header=None)
        else:
            self.uindex, self.iindex = self._generate_index()

        self.uindex.columns = ["id", "val"]
        self.iindex.columns = ["id", "val"]


    def _generate_index(self):
        print "generating user,item index..."
        user_index = {}
        item_index = {}
        user_list = self.users.USER_ID_hash.tolist()
        item_list = self.coupons.COUPON_ID_hash.tolist()
        e = 0
        for user in user_list:
            val = self.users[self.users.USER_ID_hash == user].index[0]
            user_index[user] = val
            e += 1
            if e % 1000 == 0:
                print "At user: ", e
        e = 0
        for item in item_list:
            val = self.coupons[self.coupons.COUPON_ID_hash == item].index[0] + self.num_users
            item_index[item] = val
            e += 1
            if e % 1000 == 0:
                print "At item: ", e
        uindex = pd.DataFrame.from_dict(user_index, orient='index')
        uindex.reset_index(level=0, inplace=True)
        iindex = pd.DataFrame.from_dict(item_index, orient='index')
        iindex.reset_index(level=0, inplace=True)

        print "saving index..."
        uindex.to_csv(self.uindex_fname, header=False, index=False)
        iindex.to_csv(self.iindex_fname, header=False, index=False)

        return uindex, iindex


    def convert(self):
        self._convert_train()
        self._convert_test()


    def _convert_train(self):

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

        self.result["target"] = 1.0

        print "adding negative target values..."
        neg_values = pd.DataFrame()
        neg_values["target"] = 0.0

        user_df = self.uindex[self.uindex.id.isin(self.purchases.USER_ID_hash)]
        item_df = self.iindex[self.iindex.id.isin(self.purchases.COUPON_ID_hash)]

        print "adding negative user,item indicators..."
        print "Number of users: ", user_df.shape[0]
        print "Number of items: ", item_df.shape[0]
        e = 0
        user_indicator_list = []
        item_indicator_list = []
        simil_indicator_list = []
        for user_row in user_df:
            uval = str(user_row.val) + ":1"
            for item_row in item_df:
                if self.purchases[(self.purchases.USER_ID_hash == user_row.id) & (self.purchases.COUPON_ID_hash == item_row.id)].empty:
                    # get user value
                    user_indicator_list.append(uval)
                    # get item value
                    ival = str(int(item_row.val) + self.num_users) + ":1"
                    item_indicator_list.append(ival)
                    # get similar items
                    sval = str(self.result[self.result.user == uval]["other_items"].unique()[0])
                    simil_indicator_list.append(sval)

            e += 1
            if e % 250 == 0:
                print "At user: ", e

        neg_values["user"] = pd.Series(user_indicator_list)
        neg_values["item"] = pd.Series(item_indicator_list)
        neg_values["other_items"] = pd.Series(simil_indicator_list)

        neg_values["target"] = 0.0

        self.result = self.result.append(neg_values)


    def _convert_test(self):
        user_df = self.uindex[self.uindex.id.isin(self.purchases.USER_ID_hash)]
        item_df = self.iindex[self.iindex.id.isin(self.coupons_test.COUPON_ID_hash)]

        print "converting test set..."
        print "Number of users: ", user_df.shape[0]
        print "Number of items: ", item_df.shape[0]
        e = 0
        user_indicator_list = []
        item_indicator_list = []
        simil_indicator_list = []
        for user_row in user_df:
            uval = str(user_row.val) + ":1"
            for item_row in item_df:
                # get user value
                user_indicator_list.append(uval)
                # get item value
                ival = str(int(item_row.val) + self.num_users) + ":1"
                item_indicator_list.append(ival)
                # get similar items
                temp = self.result[self.result.user == uval]
                if temp.shape[0] == 0:
                    sval = ""
                else:
                    sval = str(temp["other_items"].unique()[0])
                simil_indicator_list.append(sval)

            e += 1
            if e % 250 == 0:
                print "At user: ", e

        self.result_test["user"] = pd.Series(user_indicator_list)
        self.result_test["item"] = pd.Series(item_indicator_list)
        self.result_test["other_items"] = pd.Series(simil_indicator_list)


    def write(self, train_output_fp, test_output_fp):
        print "writing to train,test data to file..."
        self.result.to_csv(train_output_fp, sep=" ", header=False, index=False)
        self.result_test.to_csv(test_output_fp, sep=" ", header=False, index=False)


if __name__ == '__main__':

    train_output_path = sys.argv[1]
    test_output_path = sys.argv[2]

    users, coupons_train, coupons_test, purchases = load()

    libfm = LibfmLoader(users, coupons_train, coupons_test, purchases.ix[0:100], reset_index=False)
    libfm.convert()
    libfm.write(train_output_path, test_output_path)


