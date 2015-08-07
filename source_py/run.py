from source_py.data import DataLoader
from source_py.model import ContentFilter
from source_py.cv import Validator


def run(output_filename):
    load = DataLoader()
    model = ContentFilter(load.coupons_train, load.coupons_test, load.user_list, load.details_train)
    model.run()
    final_df = model.predict()
    final_df.to_csv(output_filename, sep=",", index=False, header=True)

if __name__ == '__main__':

    #run('submissions/testing.csv')

    load = DataLoader()

    # Note: training coupons span 362 days from 2011-06-27
    cv = Validator('2011-06-27', 350, 7, load)
    predictions = cv.run()
    score = cv.mapk(10, cv.actual, predictions)

    print "MAP Score: ", score





