import logging
from timeit import default_timer as timer

class Timer(object):
    units = [
        ("msec", 1e-3, 100),
        ("sec", 1, 59),
        ("min", 60, 59),
        ("hours", 60*60, 0),
    ]

    def __init__(self, msg="Elapsed time", logger=None):
        """ """
        if logger is None:
            logger = logging.debug
        self.msg = msg
        self.logger = logger
        self.dt = 0

    def messg(self, dt):
        """ """
        for name, mult, thresh in self.units:
            if dt < thresh*mult:
                break
        dt /= mult
        return "%s %g %s"%(self.msg, dt, name)

    def __enter__(self):
        """ """
        self.t0 = timer()
        return self

    def __exit__(self, *args):
        """ """
        self.dt = timer() - self.t0
        self.logger(self.messg(self.dt))


if __name__=="__main__":
    import time
    with Timer("test time:"):
        time.sleep(1)

    T = Timer()

    t = 5e-3
    for i in range(40):
        print (t, T.messg(t))
        t *= 1.5
