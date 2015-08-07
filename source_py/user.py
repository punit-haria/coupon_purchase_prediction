

class User(object):

    def __init__(self, users, index, purchases, coupons):
        """
        :param users: pandas.DataFrame of Users
        :param index: row index into user_list DataFrame for this user
        :param purchases: pandas.DataFrame of All user purchases (i.e. details_train)
        :param coupons: pandas.DataFrame of All available Coupons to match purchases to
        """
        # this user
        self.user = users.ix[index]
        self.index = index
        # transactions
        user_buys = purchases[purchases.USER_ID_hash == self.get_id()]
        # purchased coupons
        self.coupons = coupons[coupons.COUPON_ID_hash.isin(user_buys.COUPON_ID_hash)]


    def get_id(self):
        """
        :return: ID of this user
        """
        return self.user.USER_ID_hash


    @staticmethod
    def num_coupons():
        """
        :return: the number of coupons to recommend for this user
        """
        return 10


