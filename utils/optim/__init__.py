from math import sqrt
import numpy as np
from utils.logger import Logger
log = Logger('_GradientDescent')


class _GradientDescent(object):
    """Class to hold gradient descent properties"""
    def __init__(self, problem, decreasing_rate='sqrt',
                 stop='up5', tol=1e-10):
        '''Gradient Descent handeler

        Parameters
        ----------
        param: list of the Parameters
        alpha: learning rate controler
        grad: function computing the gradient
            given the current parameters
        decreasing_rate: {'sqrt', 'linear'} deacreasing rate
            for the learning rate
        '''
        self.pb = problem
        self.alpha = 4/problem.L
        self.decreasing_rate = decreasing_rate
        self.t = 0
        self.pt = np.copy(problem.x0)
        self.cost = [self.pb.cost(self.pt)]
        self.stop = stop
        self.finished = False
        self.tol = tol
        self.log_x = [np.copy(self.pt[0])]

    def __repr__(self):
        return '_GradientDescent'

    def update(self):
        '''Update the parameters given

        Parameters
        ----------
        grad: list, optional (default: None)
            list of the gradient for each parameters
        '''
        if self.finished:
            return True
        self.t += 1
        self.p_update()
        self.cost += [self.pb.cost(self.pt)]
        self.log_x += [np.copy(self.pt[0])]
        return self._stop()

    def p_update(self):
        '''Update the parameters
        '''
        grad = self.pb.grad(self.pt)
        self.p_grad = grad
        lr = self._get_lr()
        self.pt = [p-l*dp for l, p, dp in zip(lr, self.pt, grad)]

    def _get_lr(self):
        '''Auxillary funciton, return the learning rate
        '''
        lr = self.alpha
        if self.decreasing_rate == 'sqrt':
            lr /= sqrt(self.t)
        elif self.decreasing_rate == 'linear':
            lr /= self.t
        elif self.decreasing_rate == 'k2':
            lr *= 2/(self.t+2)
        elif hasattr(self.decreasing_rate, '__call__'):
            lr = self.decreasing_rate(self.t)
        return [lr]*len(self.pt)

    def _stop(self):
        '''Implement stopping criterion
        '''
        # No stopping criterions
        if self.stop == 'none':
            return False

        cost = self.cost
        lc = cost[-1]
        # If the cost move less than tol, stop
        if len(cost) > 5 and lc <= cost[-4] < lc+self.tol:
            log.info('Stop - No advance cost - {}'.format(self.__repr__()))
            self.finished = True
            return self.finished

        # If |x_n-x_n-1| < tol, stop
        if np.sum(np.abs(self.log_x[-1]-self.log_x[-2])) < self.tol:
            self.finished = True
            log.info('Stop - No advance X - {}'.format(self.__repr__()))

        # Other stopping criterion
        if self.stop == 'up5':
            self.finished = (len(cost) > 5 and cost[-5] < lc)
            if self.finished:
                log.info('Stop - Up 5 - {}'.format(self.__repr__()))
            return self.finished
        else:
            return False


class GD_Exception(Exception):
    """Gradient Descent exception"""
    def __init__(self, msg):
        super(GD_Exception, self).__init__()
        self.msg = msg
