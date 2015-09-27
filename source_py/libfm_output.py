__author__ = 'Punit'

import pandas as pd


def format():

    print "reading predictions..."

    preds = pd.read_csv("datalibfm/libfm_output.txt", header=None)
    preds.columns = ["preds"]

    print "reading dictionaries..."

    user_df = pd.read_csv("datalibfm/user_dict.txt")
    item_df = pd.read_csv("datalibfm/item_dict.txt")
    user_list = pd.read_csv("raw_data/user_list.csv")

    print "reading test index..."

    test = pd.read_csv("datalibfm/test_index.txt", header=None, sep=" ")
    test.columns = ["user_index", "item_index"]
    assert test.shape[0] == preds.shape[0]

    print "merging with user and item dictionaries..."

    test["preds"] = preds.preds
    test = test.merge(user_df, how='left', on='user_index')
    test.drop('user_index', axis=1, inplace=True)
    test = test.merge(item_df, how='left', on='item_index')
    test.drop('item_index', axis=1, inplace=True)

    print "getting top 10 probabilities..."

    grouped = test.groupby("USER_ID_hash")
    largest = grouped['preds'].nlargest(10)











if __name__ == '__main__':

    print "formatting for kaggle..."