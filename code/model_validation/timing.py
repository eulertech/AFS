from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def timing(logger=None, callback='No callback message'):
    timer = _Timer()

    yield timer

    if logger is not None:
        callback += ' - %.2f seconds'
        logger.debug(callback % timer.age)

class _Timer(object):

    def __init__(self):
        self._start, self._elapsed = self.now, self.now

    @property
    def now(self):
        return default_timer()    

    @property
    def start(self):
        return self._start

    @property
    def elapsed(self):
        now = self.now
        elapsed = now - self._elapsed
        self._elapsed = now
        return elapsed

    @property
    def age(self):
        return self.now - self.start