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
    visits = pd.read_csv("raw_data/coupon_visit_train.csv")

    return user_list, coupons_train, coupons_test, details_train, visits

class LibfmLoader(object):

    def __init__(self, users, coupons_train, coupons_test, purchases, visits, reset_index=False):
        self.users = users
        self.coupons_train = coupons_train
        self.coupons_test = coupons_test
        self.purchases = purchases

        self.coupons_train["type"] = "train"
        self.coupons_test["type"] = "test"
        self.coupons = self.coupons_train.append(self.coupons_test)
        self.coupons.reset_index(inplace=True)

        self.visits = visits[visits.VIEW_COUPON_ID_hash.isin(self.coupons.COUPON_ID_hash)]

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
            uindex = pd.read_csv(self.uindex_fname, header=None)
            iindex = pd.read_csv(self.iindex_fname, header=None)
            uindex.columns = ["id", "val"]
            iindex.columns = ["id", "val"]
            self.uindex = uindex.set_index('id')['val'].to_dict()
            self.iindex = iindex.set_index('id')['val'].to_dict()
        else:
            self.uindex, self.iindex = self._generate_index()


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

        print "saving index..."
        uindex = pd.DataFrame.from_dict(user_index, orient='index')
        uindex.reset_index(level=0, inplace=True)
        iindex = pd.DataFrame.from_dict(item_index, orient='index')
        iindex.reset_index(level=0, inplace=True)
        uindex.to_csv(self.uindex_fname, header=False, index=False)
        iindex.to_csv(self.iindex_fname, header=False, index=False)

        return user_index, item_index


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
            purchased_set_len = len(purchased_set)
            if purchased_set_len == 0: return fstr
            value = str(1.0 / math.sqrt(len(purchased_set)))
            for item in purchased_set:
                fstr += str(self.coupons[self.coupons.COUPON_ID_hash == item].index[0] + self.num_users + self.num_items)
                fstr += ":" + value + " "
            return fstr

        self.result["other_items"] = self.result.apply(similar_items_indicator, axis=1)

        self.result["target"] = 1.0

        print "negative user,item indicators..."

        unpvis =  self.visits[self.visits.PURCHASE_FLG == 0]
        unpvis = unpvis[["USER_ID_hash", "VIEW_COUPON_ID_hash"]].drop_duplicates()
        unpvis.columns = ["USER_ID_hash", "COUPON_ID_hash"]

        print "adding", unpvis.shape[0], "indicators..."

        neg_values = pd.DataFrame()
        neg_values["target"] = 0.0
        neg_values["user"]  = unpvis.apply(user_indicator, axis=1)
        neg_values["item"] = unpvis.apply(coupon_indicator, axis=1)
        neg_values["other_items"] = neg_values.apply(similar_items_indicator, axis=1)

        print "reading probabilities of purchase (based on visits)..."
        prob_purchase = pd.read_csv("datalibfm/prob_purchase.txt")

        print "finding corresponding user indices..."

        prob_purchase["user"] = prob_purchase.apply(user_indicator, axis=1)
        prob_purchase = prob_purchase[["user", "PROB_PURCHASE"]]

        print "adding negative targets..."

        neg_values = neg_values.merge(prob_purchase, how='left', on='user')
        neg_values["target"] = neg_values["PROB_PURCHASE"]
        neg_values = neg_values[["target","user","item","other_items"]]
        neg_values["target"].fillna(0.0)

        self.result = self.result.append(neg_values)


    def _convert_test(self):
        user_keyset = self.users[self.users.USER_ID_hash.isin(self.purchases.USER_ID_hash)]["USER_ID_hash"].tolist()
        item_keyset = self.coupons_test.COUPON_ID_hash.tolist()

        print "converting test set..."
        print "Number of users: ", len(user_keyset)
        print "Number of items: ", len(item_keyset)
        e = 0
        user_indicator_list = []
        item_indicator_list = []
        simil_indicator_list = []
        for user_id in user_keyset:
            uval = str(self.uindex[user_id]) + ":1"
            for item_id in item_keyset:
                # get user value
                user_indicator_list.append(uval)
                # get item value
                ival = str(int(self.iindex[item_id]) + self.num_users) + ":1"
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

        self.result_test["target"] = 0.0
        self.result_test["user"] = pd.Series(user_indicator_list)
        self.result_test["item"] = pd.Series(item_indicator_list)
        self.result_test["other_items"] = pd.Series(simil_indicator_list)
        self.result_test["target"] = 0.0


    def write(self, train_output_fp, test_output_fp):
        print "writing to train,test data to file..."
        self.result.to_csv(train_output_fp, sep=" ", header=False, index=False)
        self.result_test.to_csv(test_output_fp, sep=" ", header=False, index=False)


if __name__ == '__main__':

    users, coupons_train, coupons_test, purchases, visits = load()

    libfm = LibfmLoader(users, coupons_train, coupons_test, purchases, visits, reset_index=False)
    libfm.convert()

    train_output_path = sys.argv[1]
    test_output_path = sys.argv[2]
    libfm.write(train_output_path, test_output_path)


