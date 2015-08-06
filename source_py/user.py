from source_py.data import DataLoader
from source_py.item import ItemProfile

class User(object):

    def __init__(self, loader, index, item_profile):
        """
        :param loader: DataLoader object with access to all data
        :param index: row index into user_list DataFrame for this user
        :param item_profile : ItemProfile object
        """
        assert isinstance(loader, DataLoader)
        self.data = loader
        # this user
        self.user = self.data.user_list.ix[index]
        self.index = index
        # transactions
        purchases = self.data.details_train[self.data.details_train.USER_ID_hash == self.user.USER_ID_hash]
        # purchased coupons
        self.coupons = self.data.coupons_train[self.data.coupons_train.COUPON_ID_hash.isin(purchases.COUPON_ID_hash)]
        # item profile object
        assert isinstance(item_profile, ItemProfile)
        self.item_profile = item_profile


    def get_id(self):
        """
        :return: ID of this user
        """
        return self.user.USER_ID_hash


    def recommend(self):
        """
        :return: gets the recommended coupons from the test set for this user
        """
        # get similarity scores for each test coupon
        scores = self.item_profile.similarity(self.coupons.index, self.data.coupons_test.index)
        # x-axis: test coupons, y-axis: user coupons
        scores = scores.transpose()
        # compute mean similarity score for each test coupon
        scores["mean"] = scores.mean(axis=1)
        # sort by descending order of mean score
        scores.sort(columns="mean", ascending=False, inplace=True)
        # get top test coupon indices
        top = User.num_coupons()
        if top < 1:
            return ""
        top_indices = scores.head(n=top).index
        # get top coupon IDs
        coups = self.data.coupons_test.ix[top_indices].COUPON_ID_hash.tolist()
        # return as space-delimited string
        ids = ""
        for value in coups:
            ids += value + " "
        return ids


    @staticmethod
    def num_coupons():
        """
        :return: the number of coupons to recommend for this user
        """
        return 5


