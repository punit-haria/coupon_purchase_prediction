from numpy.linalg import norm


class ItemProfile(object):

    def __init__(self, loader, simfn="cosine"):
        """
        :param loader: DataLoader
        :param simfn : similarity function
        """
        self.data = loader
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
        train = self.data.coupons_train.drop("COUPON_ID_hash", 1)
        test = self.data.coupons_test.drop("COUPON_ID_hash", 1)

        # normalize data
        train = train.div(train.apply(norm, axis=1), axis='index')
        test = test.div(test.apply(norm, axis=1), axis='index')

        # compute cosine similarities
        return train.dot(test.transpose())

