from source_py.data import DataLoader
from source_py.model import ContentFilter
from source_py.cv import Validator

import numpy as np
from random import randint


def run(output_filename):
    load = DataLoader()
    model = ContentFilter(load.coupons_train, load.coupons_test,
                          load.user_list, load.details_train)
    model.run()
    final_df = model.predict()
    final_df.to_csv(output_filename, sep=",", index=False, header=True)


def randomize():
    start_offset = randint(0,50)
    training_period = randint(200,354 - start_offset)
    day_zero = np.datetime64('2011-06-27')
    start = str(day_zero + start_offset)
    validation_period = 7

    return start, training_period, validation_period


def validate(output_filename):
    load = DataLoader()

    # Note: training coupons span 362 days from 2011-06-27

    #start = '2011-06-27'
    #training_period = 350
    #validation_period = 7
    start, training_period, validation_period = randomize()

    config = "Split:\n"
    config += "Start date: " + start + "\n"
    config += "End date: " + str(np.datetime64(start) + training_period) + "\n"
    config += "Training period: " + str(training_period) + "\n"
    config += "Validation period: " + str(validation_period) + "\n\n"

    print config

    cv = Validator(start, training_period, validation_period, load)
    predictions = cv.run()
    score = cv.mapk(10, cv.actual, predictions)

    print "MAP Score: ", score

    config += cv.model.get_configuration()
    config += "\n\nMAP Score: " + str(score)
    with open(output_filename,'w') as f:
        f.write(config)


if __name__ == '__main__':

    #run('submissions/submission.csv')

    load = DataLoader()

    #validate("selection/model_config.txt")







