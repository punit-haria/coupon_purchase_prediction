__author__ = 'punit'

import time

class Timer(object):
    """
    Timer class that functions as a stopwatch to measure times
    for sequences of functions.
    """

    def __init__(self):
        self.times = []
        self.seq = []
        self.t0 = None
        self.t1 = None

    def start(self):
        """
        Clears timer value and starts from 0.
        """
        self.t0 = time.time()

    def stop(self):
        """
        Stops running timer. Adds value to sequence.
        """
        self.t1 = time.time()
        self.seq.append(self.t1 - self.t0)

    def stopstart(self):
        """
        Stops timer. Adds value to sequence. Starts new timer.
        """
        self.stop()
        self.start()

    def save(self):
        """
        Saves a sequence of times to list.
        """
        self.times.append(self.seq)
        self.seq = []



