import numpy as np
from multiprocessing import Process

from .solver import Solver
from utils.logger import Logger
log = Logger('WorkerSolver', 20)


class WorkerSolver(Process):
    def __init__(self, qin, qout, id_w=0, seed=None, **param):
        self.qin = qin
        self.qout = qout
        self.solver = Solver(**param)
        self.id = id_w
        self.seed = seed
        super(WorkerSolver, self).__init__()

    def run(self):
        seed = self.seed
        if seed is None:
            seed = np.random.randint(214783648)
        np.random.seed(seed)
        idp, pb = self.qin.get()
        while idp is not None:
            log.debug('Worker {} - |qin| = {}'
                      ''.format(self.id, self.qin.qsize()))
            solution = self.solver.solve(pb, name='Sig_{}'.format(idp))
            self.qout.put((idp, solution, pb.cost(solution)))
            idp, pb = self.qin.get()
        log.debug('Worker {} finished'.format(self.id))
        return 0
