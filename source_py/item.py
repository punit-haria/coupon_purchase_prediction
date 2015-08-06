from numpy.linalg import norm


class ItemProfile(object):

    def __init__(self, loader, simfn="cosine", train_only=False):
        """
        :param loader: DataLoader
        :param simfn : similarity function
        """
        self.data = loader
        self.matrix = None
        self.train = train_only
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
        left = self.data.coupons_train.drop("COUPON_ID_hash", 1)
        if self.train:
            right = self.data.coupons_train.drop("COUPON_ID_hash", 1)
        else:
            right = self.data.coupons_test.drop("COUPON_ID_hash", 1)

        # normalize data
        left = left.div(left.apply(norm, axis=1), axis='index')
        right = right.div(right.apply(norm, axis=1), axis='index')

        # compute cosine similarities
        return left.dot(right.transpose())

