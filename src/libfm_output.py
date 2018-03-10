import pandas as pd
import sys

# Lib-FM Output


def format_kaggle(output_fname):

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

    val = test.groupby("USER_ID_hash")["preds"].nlargest(10)

    test.reset_index(inplace=True)
    test.rename(columns={'index':'new_index'}, inplace=True)

    val = val.reset_index(level=1, inplace=False)
    val.columns = ["new_index", "preds"]

    test = val.merge(test, how='left', on='new_index')
    test.drop(['new_index','preds_y'], axis=1, inplace=True)
    test.rename(columns={'preds_x':'preds'}, inplace=True)

    print "formatting data for kaggle..."

    def kfmt(x):
        return " ".join(x.COUPON_ID_hash)

    test = test.groupby("USER_ID_hash").aggregate(kfmt)
    test.drop('preds', axis=1, inplace=True)
    test.reset_index(inplace=True)
    test.rename(columns={'COUPON_ID_hash':'PURCHASED_COUPONS'}, inplace=True)

    assert test.USER_ID_hash.unique().shape[0] == user_list.shape[0]
    assert test.shape[0] == user_list.shape[0]

    print "writing..."

    test[['USER_ID_hash','PURCHASED_COUPONS']].to_csv(output_fname, sep=",", index=False, header=True)

if __name__ == '__main__':

    print "formatting for kaggle..."
    format_kaggle(sys.argv[1])















