from time import time


from toolboxTom.logger import Logger
from . import _GradientDescent
log = Logger(name='Solver', levl=20)


class Solver(object):
    """Encode a signal in the convolutional dictionary"""
    def __init__(self, optim=_GradientDescent, max_time=None,
                 i_max=1e6, debug=0, **kwargs):
        self.optim = optim
        self.param = kwargs
        self.i_max = i_max
        self.max_time = max_time
        if debug:
            log.set_level(10)

    def solve(self, pb, **kwargs):
        self.pb = pb
        self.param.update(**kwargs)
        solver = self.optim(self.pb, **self.param)
        finished = False
        self.start_time = time()
        self.iter = 0
        while not finished and not self._stop():
            finished = solver.update()
            self.iter += 1
        self._end()
        self.pt = solver.pt
        self.cost = solver.cost
        return self.pt

    def _stop(self):
        t = time() - self.start_time
        n, m = self.iter, self.i_max
        if (self.max_time is not None and
                t/self.max_time > n/m):
            n, m = t, self.max_time
        if n >= m:
            return True
        log.progress(levl=10, name='ConvSparseCoder', iteration=n,
                     i_max=m)
        return False

    def _end(self):
        log.progress(levl=10, name='ConvSparseCoder', iteration=1,
                     i_max=1)
        t = time() - self.start_time
        log.info('Time: {:.3}s; Iteration: {}'
                 ''.format(t, self.iter))
