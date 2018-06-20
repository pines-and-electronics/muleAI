import time


def regulate(iterable, hertz):
    ''' Regulates iteration

        Arguments

        iterable: 
            iterable object

        hertz: float
            iterations per second after regulation


        Note
            if the loop body takes longer than the regulation time
            interval, then nothing is done
    '''
    time_interval = 1 / hertz

    mark = time.time()

    for i in iterable:
        time_taken = time.time() - mark

        if time_taken < time_interval:
            time.sleep(time_interval - time_taken)

        mark = time.time()

        yield i

