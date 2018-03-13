from numpy.linalg import norm

import pandas as pd

from .data import DataLoader

def test():
    load = DataLoader()
    return ItemMatrix(load.coupons_train, load.coupons_test)

class ItemMatrix(object):
    """
    Constructs an Item-Item similarity matrix between training and test coupons.
    """

    def __init__(self, train_coupons, test_coupons):
        """
        :param train_coupons: pandas.DataFrame of training coupon data
        :param test_coupons: pandas.DataFrame of test coupon data
        """
        self.fields = ["COUPON_ID_hash",
                       "CAPSULE_TEXT", "GENRE_NAME",
                       "PRICE_RATE", "CATALOG_PRICE", "DISCOUNT_PRICE",
                       "VALIDPERIOD",
                       "USABLE_DATE_MON", "USABLE_DATE_TUE", "USABLE_DATE_WED",
                       "USABLE_DATE_THU", "USABLE_DATE_FRI", "USABLE_DATE_SAT",
                       "USABLE_DATE_SUN", "USABLE_DATE_HOLIDAY", "USABLE_DATE_BEFORE_HOLIDAY",
                       "large_area_name", "ken_name", "small_area_name"]

        # keep relevant coupon fields only
        self.train_coupons = train_coupons[self.fields].copy(deep=True)
        self.test_coupons = test_coupons[self.fields].copy(deep=True)

        # expand categorical variables
        self.categorical = ["CAPSULE_TEXT", "GENRE_NAME",
                            "large_area_name", "ken_name", "small_area_name"]
        self.categorical_weights = [1.0, 1.0,
                                    1.0, 1.0, 1.0]
        self._expand(transform=0.0)

        # scale numerical variables
        self.numerical = ["PRICE_RATE", "CATALOG_PRICE", "DISCOUNT_PRICE",
                       "VALIDPERIOD",
                       "USABLE_DATE_MON", "USABLE_DATE_TUE", "USABLE_DATE_WED",
                       "USABLE_DATE_THU", "USABLE_DATE_FRI", "USABLE_DATE_SAT",
                       "USABLE_DATE_SUN", "USABLE_DATE_HOLIDAY", "USABLE_DATE_BEFORE_HOLIDAY"]
        self.numerical_weights = [1.0, 1.0, 1.0,
                        1.0,
                        1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0]
        self._scale(transform=0.0)

        # replace missing values
        self._replace_nan(0.)

        # item-item similarity matrix
        self.matrix = self._generate_cosine()


    def similarity(self, train_range, test_range):
        """
        :param train_range, test_range : lists of indices into training and test coupon DataFrames
        :return the cosine similarities of these coupons with the input coupons
        """
        return self.matrix.ix[train_range][test_range]


    def _generate_cosine(self):
        """
        Generates an Item-Item cosine similarity matrix between the training and test coupons.
        """
        left = self.train_coupons.drop("COUPON_ID_hash", 1)
        right = self.test_coupons.drop("COUPON_ID_hash", 1)

        # normalize data
        left = left.div(left.apply(norm, axis=1), axis='index')
        right = right.div(right.apply(norm, axis=1), axis='index')

        # compute cosine similarities
        return left.dot(right.transpose())


    def _expand(self, transform):
        """
        Expands categorical variables in training and test sets into
        0-1 dummy variables, scales them according to input weights, and
        finally transforms upwards slightly.
        No future information is introduced from the test set.
        """
        # need to concatenate to maintain column consistency when expanding
        self.train_coupons["type"] = "train"
        self.test_coupons["type"] = "test"
        merged = self.train_coupons.append(self.test_coupons)

        # expand and scale categorical variables
        for field, weight in zip(self.categorical, self.categorical_weights):
            df = (weight * pd.get_dummies(merged[field])) + transform
            merged = pd.concat([merged, df], axis=1)

        # drop original categoricals
        merged.drop(self.categorical, axis=1, inplace=True)

        # split back into training and test sets
        self.train_coupons = pd.DataFrame.copy(merged[merged.type == "train"], deep=True)
        self.train_coupons.reset_index(inplace=True)
        self.train_coupons.drop(["index","type"], axis=1, inplace=True)

        self.test_coupons = pd.DataFrame.copy(merged[merged.type == "test"], deep=True)
        self.test_coupons.reset_index(inplace=True)
        self.test_coupons.drop(["index","type"], axis=1, inplace=True)

        # validate coupon expansion
        assert len(self.train_coupons.columns) == len(self.test_coupons.columns)
        for left, right in zip(self.train_coupons.columns, self.test_coupons.columns):
            assert left == right



    def _scale(self, transform):
        """
        Normalizes all variables in the training and test sets using a set of weights.
        Variables are then transformed upwards slightly.
        Note: NO future information is introduced. Test set normalization
        is done using training set min/max values.
        """
        # scale between 0 and 1
        df = self.train_coupons[self.numerical]
        train_min = df.min()
        train_max = df.max()
        self.train_coupons[self.numerical] = (df - train_min) / (train_max - train_min)
        df = self.test_coupons[self.numerical]
        self.test_coupons[self.numerical] = (df - train_min) / (train_max - train_min)

        # apply weights and transform
        for field, weight in zip(self.numerical, self.numerical_weights):
            self.train_coupons[field] = (weight * self.train_coupons[field]) + transform
            self.test_coupons[field] = (weight * self.test_coupons[field]) + transform


    def _replace_nan(self, value):
        """
        Replaces null/nan values.
        """
        self.train_coupons.fillna(value=value, inplace=True)
        self.test_coupons.fillna(value=value, inplace=True)





