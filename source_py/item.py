from numpy.linalg import norm
import pandas as pd


class ItemProfile(object):

    def __init__(self, train, test, simfn="cosine"):
        """
        :param train/test : pandas.DataFrame for training and test coupon data
        :param simfn : similarity function
        """
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        self.train = train
        self.test = test
        self.matrix = None
        if simfn == "cosine":
            self.simfn = self._generate_cosine
        else:
            raise NotImplementedError


    def similarity(self, train_range, test_range):
        """
        :param train_range, test_range : lists of indices into training and test coupon DataFrames
        :return the cosine similarities of these coupons with the input coupons
        """
        simil_matrix = self.generate()
        return simil_matrix.ix[train_range][test_range]


    def generate(self):
        """
        Returns the Item-Item similarity matrix between training and test coupons.
        """
        if self.matrix is None:
            self.matrix = self.simfn()

        return self.matrix


    def _generate_cosine(self):
        """
        Generates an Item-Item cosine similarity matrix between the training and test coupons.
        """
        left = self.train.drop("COUPON_ID_hash", 1)
        right = self.test.drop("COUPON_ID_hash", 1)

        # normalize data
        left = left.div(left.apply(norm, axis=1), axis='index')
        right = right.div(right.apply(norm, axis=1), axis='index')

        # compute cosine similarities
        return left.dot(right.transpose())
