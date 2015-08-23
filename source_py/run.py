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
    final_df = model.predict
    final_df.to_csv(output_filename, sep=",", index=False, header=True)

    return model


def randomize():
    start_offset = randint(0,50)
    training_period = randint(200,354 - start_offset)
    day_zero = np.datetime64('2011-06-27')
    start = str(day_zero + start_offset)
    validation_period = 7

    return start, training_period, validation_period


def validate(start, training_period, validation_period, output):
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

    output.put([cv.score, config])


def parallel_validate(output_file):
    # Note: training coupons span 362 days from 2011-06-27

    output = mp.Queue()

    processes = []
    start, training_period, validation_period = randomize()
    processes.append(mp.Process(target=validate, args=(
                                      start, training_period, validation_period,output)))
    start, training_period, validation_period = randomize()
    processes.append(mp.Process(target=validate, args=(
                                      start, training_period, validation_period,output)))
    start, training_period, validation_period = randomize()
    processes.append(mp.Process(target=validate, args=(
                                      start, training_period, validation_period,output)))
    start, training_period, validation_period = randomize()
    processes.append(mp.Process(target=validate, args=(
                                      start, training_period, validation_period,output)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print "Processes complete!"

    res = [output.get() for p in processes]

    mapscores = []
    config = ""
    for tup in res:
        mapscores.append(tup[0])
        config += tup[1] + "\n\n----------------------------\n\n"

    config += "FINAL MAP SCORE: " + str(np.array(mapscores).mean())

    print config

    with open(output_file,'w') as f:
        f.write(config)




if __name__ == '__main__':

    #model = run('submissions/submission.csv')

    parallel_validate('selection/model_config_8.txt')














