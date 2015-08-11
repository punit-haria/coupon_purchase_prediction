from source_py.data import DataLoader
from source_py.model import Model
from source_py.cv import Validator

import numpy as np
from random import randint
import multiprocessing as mp


def run(output_filename):
    load = DataLoader()
    model = Model(load.coupons_train, load.coupons_test,
                          load.user_list, load.details_train)
    model.run()
    final_df = model.predict()
    final_df.to_csv(output_filename, sep=",", index=False, header=True)

    return model


def randomize():
    start_offset = randint(0,50)
    training_period = randint(200,354 - start_offset)
    day_zero = np.datetime64('2011-06-27')
    start = str(day_zero + start_offset)
    validation_period = 7

    return start, training_period, validation_period


def validate(output_filename, start, training_period, validation_period, output):
    load = DataLoader()

    config = "Split:\n"
    config += "Start date: " + start + "\n"
    config += "End date: " + str(np.datetime64(start) + training_period) + "\n"
    config += "Training period: " + str(training_period) + "\n"
    config += "Validation period: " + str(validation_period) + "\n\n"

    print config

    cv = Validator(start, training_period, validation_period, load)
    predictions = cv.run()
    cv.score = cv.mapk(10, cv.actual, predictions)

    print "MAP Score: ", cv.score

    config += cv.model.get_configuration()
    config += "\n\nMAP Score: " + str(cv.score)
    with open(output_filename,'w') as f:
        f.write(config)

    output.put([cv.score, config])


def parallel_validate():
    #start, training_period, validation_period = ('2011-06-27', 354, 7)
    #start, training_period, validation_period = randomize()

    # Note: training coupons span 362 days from 2011-06-27

    output = mp.Queue()

    processes = []
    start, training_period, validation_period = randomize()
    processes.append(mp.Process(target=validate, args=('selection/model_config_5.1.txt',
                                      start, training_period, validation_period,output)))
    start, training_period, validation_period = randomize()
    processes.append(mp.Process(target=validate, args=('selection/model_config_5.2.txt',
                                      start, training_period, validation_period,output)))
    start, training_period, validation_period = randomize()
    processes.append(mp.Process(target=validate, args=('selection/model_config_5.3.txt',
                                      start, training_period, validation_period,output)))
    start, training_period, validation_period = randomize()
    processes.append(mp.Process(target=validate, args=('selection/model_config_5.4.txt',
                                      start, training_period, validation_period,output)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print "Finished processing!"

    return [output.get() for p in processes]


if __name__ == '__main__':

    #model = run('submissions/submission.csv')

    res = parallel_validate()









