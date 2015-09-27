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

        self.user_df = None
        self.item_df = None
        self.rated = None

        self.result = None
        self.result_test = None

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

        print "adding positive examples..."

        # libfm notation conversion function
        def libfm_notation(val):
            return str(val)+":1"

        print "defining user-index mapping..."

        user_df = self.users.reset_index(level=0, inplace=False)
        user_df.rename(columns={'index':'user_index'}, inplace=True)
        user_df = user_df[["user_index", "USER_ID_hash"]]
        user_df["user_index"] = user_df.user_index.apply(libfm_notation)

        print "adding user indicators..."

        plhold = self.purchases[["USER_ID_hash", "COUPON_ID_hash"]].copy(deep=True)
        plhold = plhold.merge(user_df, how='left', on='USER_ID_hash')

        print "defining item-index and similar item to index mapping..."

        item_df = self.coupons.reset_index(level=0, inplace=False)
        item_df.rename(columns={'index':'item_index'}, inplace=True)
        item_df = item_df[["item_index", "COUPON_ID_hash"]]
        item_df["item_index"] = item_df.item_index + self.num_users
        item_df["simil_item_index"] = item_df["item_index"] + self.num_items
        item_df["item_index"] = item_df.item_index.apply(libfm_notation)
        item_df["simil_item_index"] = item_df.simil_item_index.apply(libfm_notation)

        print "adding item indicators..."

        plhold = plhold.merge(item_df, how='left', on='COUPON_ID_hash')
        plhold = plhold[["USER_ID_hash", "COUPON_ID_hash", "user_index", "item_index"]]

        print "defining similar item indicators..."

        rated = self.visits.rename(columns={'VIEW_COUPON_ID_hash':'COUPON_ID_hash'}, inplace=False)
        rated = rated[["USER_ID_hash", "COUPON_ID_hash"]]
        rated = rated.append(self.purchases[["USER_ID_hash", "COUPON_ID_hash"]])
        rated.drop_duplicates(inplace=True)
        rated = rated.merge(item_df, how='left', on='COUPON_ID_hash')
        rated = rated[["USER_ID_hash", "simil_item_index"]]
        rated["simil_item_index"] += " "
        rated = rated.groupby("USER_ID_hash").aggregate(lambda x: " ".join(x.simil_item_index))
        rated.reset_index(level=0, inplace=True)

        print "adding similar item indicators..."

        plhold = plhold.merge(rated, how='left', on='USER_ID_hash')

        print "finalizing positive examples..."

        plhold["target"] = 1.0
        plhold = plhold[["USER_ID_hash", "COUPON_ID_hash", "target", "user_index", "item_index", "simil_item_index"]]

        print "adding negative examples..."

        unpvis =  self.visits[self.visits.PURCHASE_FLG == 0]
        unpvis = unpvis[["USER_ID_hash", "VIEW_COUPON_ID_hash"]].drop_duplicates()
        unpvis.rename(columns={'VIEW_COUPON_ID_hash':'COUPON_ID_hash'}, inplace=True)

        print "adding user indicators..."

        unpvis = unpvis.merge(user_df, how='left', on='USER_ID_hash')

        print "adding item indicators..."

        unpvis = unpvis.merge(item_df, how='left', on='COUPON_ID_hash')
        unpvis = unpvis[["USER_ID_hash", "COUPON_ID_hash", "user_index", "item_index"]]

        print "adding similar item indicators..."

        unpvis = unpvis.merge(rated, how='left', on='USER_ID_hash')

        print "reading probabilities of purchase (based on visits)..."

        prob_purchase = pd.read_csv("datalibfm/prob_purchase.txt")
        prob_purchase = prob_purchase[["USER_ID_hash", "PROB_PURCHASE"]]

        print "finalizing negative examples..."

        unpvis = unpvis.merge(prob_purchase, how='left', on='USER_ID_hash')
        unpvis.rename(columns={'PROB_PURCHASE':'target'}, inplace=True)
        unpvis = unpvis[["USER_ID_hash", "COUPON_ID_hash", "target", "user_index", "item_index", "simil_item_index"]]

        print "finalizing training set..."

        self.user_df = user_df
        self.item_df = item_df
        self.rated = rated
        self.result = plhold.append(unpvis)


    def _convert_test(self):

        print "generating test examples with user and item indicators..."

        test_userdf = self.user_df.copy(deep=True)
        test_userdf["temp"] = 1
        test_itemdf = self.item_df[["COUPON_ID_hash", "item_index"]]
        test_itemdf = test_itemdf[test_itemdf.COUPON_ID_hash.isin(self.coupons_test.COUPON_ID_hash)].copy(deep=True)
        test_itemdf["temp"] = 1
        testrset = pd.merge(test_userdf, test_itemdf, on='temp')[["USER_ID_hash","user_index","COUPON_ID_hash","item_index"]]

        print "adding similar item indicators..."

        testrset = testrset.merge(self.rated, how='left', on='USER_ID_hash')

        print "adding garbage target value (required by libFM)..."

        testrset["target"] = -1.0

        print "finalizing test examples..."

        testrset = testrset[["USER_ID_hash", "COUPON_ID_hash", "target", "user_index", "item_index", "simil_item_index"]]
        testrset.simil_item_index.fillna("", inplace=True)
        self.result_test = testrset


    def write(self, train_output_fp, test_output_fp):
        print "writing to train,test data to file..."
        self.result.to_csv(train_output_fp, sep=" ", header=False, index=False)
        self.result_test.to_csv(test_output_fp, sep=" ", header=False, index=False)


if __name__ == '__main__':

    print "loading data..."
    users, coupons_train, coupons_test, purchases, visits = load()

    print "initializing..."
    libfm = LibfmLoader(users, coupons_train, coupons_test, purchases, visits, reset_index=False)
    print "converting data..."
    libfm.convert()

    train_output_path = sys.argv[1]
    test_output_path = sys.argv[2]
    libfm.write(train_output_path, test_output_path)


