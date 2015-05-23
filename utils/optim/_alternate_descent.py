import numpy as np
from math import sqrt
from time import time

from utils.logger import Logger
log = Logger('_GD', 10)

from . import get_log_rate, _GradientDescent


class _AlternateDescent(object):
    """Class to hold gradient descent properties"""

    id_gd = 0

    def __init__(self, problem, decreasing_rate='sqrt',
                 stop='', tol=1e-10, graphical_cost=None,
                 name=None, debug=0, logging=False,
                 log_rate='log', i_max=1000, t_max=40,
                 methods=[]):
        '''Gradient Descent handeler

        Parameters
        ----------
        problem: _CoupledProblem to solve
        param: list of the Parameters
        alpha: learning rate controler
        grad: function computing the gradient
            given the current parameters
        decreasing_rate: {'sqrt', 'linear'} deacreasing rate
            for the learning rate
        methods: list of optim solver and params
        '''

        self.id = _AlternateDescent.id_gd
        _AlternateDescent.id_gd += 1

        if debug > 0:
            log.set_level(10)
            log.debug('Set debug mode on')
        self.pb = problem
        self.alpha = 1/problem.L
        self.decreasing_rate = decreasing_rate
        self.stop = stop
        self.finished = False
        self.tol = tol
        self.iteration = 0
        self.t = 0

        # Logging system
        self.logging = logging
        if self.logging:
            self.log_rate = get_log_rate(log_rate)
        else:
            self.log_rate = get_log_rate('none')
        self.next_log = self.log_rate(self.iteration)
        self.i_max = i_max
        self.t_max = t_max

        self.name = name if name is not None else '_GD' + str(self.id)
        self.graph_cost = None
        if graphical_cost is not None:
            self.graph_cost = dict(name=graphical_cost, curve=self.name)
        if len(methods) < self.pb.n_param:
            methods += [(_GradientDescent, {})]*(self.pb.n_param-len(methods))
        self.solvers = [met(pb, **kwargs) for (met, kwargs), pb in
                        zip(methods, self.pb.pbs)]

    def __repr__(self):
        return self.name

    def _init_algo(self):
        for s in self.solvers:
            s._init_algo()

    def update(self):
        '''Wrap up the update with the log system
        '''
        if self.finished:
            return True
        if self.iteration == 0:
            self.start()
        self.iteration += 1
        dz = 0
        for s in self.solvers:
            dz += self.s.p_update()
        self.t = time() - self.t_start
        if self.iteration >= self.next_log:
            log.log_obj(name='cost'+str(self.id), obj=self.pb.cost(),
                        iteration=self.iteration+1, graph_cost=self.graph_cost,
                        time=self.t)
            self.next_log = self.log_rate(self.iteration)
        stop = self._stop(dz)
        if stop:
            self.end()
        return stop

    def _stop(self, dz):
        '''Implement stopping criterion
        '''

        if self.iteration >= self.i_max or self.t >= self.t_max:
            self.finished = True
            return True

        if self.stop == 'none':
            return False

        # If |x_n-x_n-1| < tol, stop
        if dz < self.tol:
            self.finished = True
            log.info('Stop - No advance X - {}'.format(self.__repr__()))
            return self.finished

    def start(self):
        log.info('Start', self)
        self.t0 = time()
        self._init_algo()
        if self.logging:
            log.log_obj(name='cost'+str(self.id), obj=np.copy(self.pt),
                        iteration=self.iteration+1, fun=self.pb.cost,
                        graph_cost=self.graph_cost, time=time()-self.t_start)
        self.t_start = time()

    def end(self):
        if self.logging:
            log.log_obj(name='cost'+str(self.id), obj=self.pt,
                        iteration=self.iteration+1, fun=self.pb.cost,
                        graph_cost=self.graph_cost, time=self.t)
        log.info('End for {}'.format(self),
                 'iteration {}, time {:.4}s'.format(self.iteration, self.t))
        log.info('Total time: {:.4}s'.format(time()-self.t0))
