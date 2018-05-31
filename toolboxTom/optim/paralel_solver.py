from numpy import ndarray
from multiprocessing import Queue
from .worker_solver import WorkerSolver
from .problem import _Problem

from toolbox.logger import Logger
log = Logger('//Solver', 20)


class ParalelSolver(object):
    """Paralell sparse coding"""
    def __init__(self, n_jobs=4, debug=0, **kwargs):
        super(ParalelSolver, self).__init__()
        self.n_jobs = n_jobs
        self.param = kwargs
        if debug:
            log.set_level(10)
            debug -= 1
        self.debug = debug

    def solve(self, problems, **kwargs):
        if type(problems) not in [list, ndarray]:
            problems = [problems]
        assert issubclass(type(problems[0]), _Problem), (
            'ParalelSolver argument is not a _Problem subclass')
        qin = Queue()
        qout = Queue()
        for i, pb in enumerate(problems):
            qin.put((i, pb))

        slaves = []
        for i in range(self.n_jobs):
            slaves += [WorkerSolver(qin, qout, id_w=i,
                                    debug=self.debug,
                                    **self.param)]
            qin.put((None, None))
            slaves[-1].start()

        # Join loop
        N_iter = len(problems)
        self.solutions = [0]*N_iter
        self.scores = [0]*N_iter
        for i in range(N_iter):
            idp, z, s = qout.get()
            self.solutions[idp] = z
            self.scores[idp] = s
            log.progress(name='Solver', iteration=i+1, i_max=N_iter)

        for s in slaves:
            s.join()
        self.problems = problems
        return self.solutions
