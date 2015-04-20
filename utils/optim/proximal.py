import numpy as np


from . import _GradientDescent
from utils.logger import Logger

log = Logger(name='ProximalDescent')


class ProximalDescent(_GradientDescent):
    """Gradient Descent with the nesterov momentum"""
    def __init__(self, problem, decreasing_rate='', f_theta='k2',
                 restart=False, debug=0, **kwargs):
        self.restart = restart
        if debug > 0:
            debug -= 1
            log.set_level(10)
        super(ProximalDescent, self).__init__(
            problem, decreasing_rate=decreasing_rate,
            debug=debug, **kwargs)
        self.theta = [1, 1]
        if type(f_theta) is float:
            self.theta = [1-f_theta]*2
        self.p_grad = np.zeros(self.pt.shape)
        self.f_theta = f_theta
        self.alpha = 1/self.pb.L

    def __repr__(self):
        return ('Proximal Descent' +
                (' with Restart' if self.restart else ''))

    def p_update(self):
        '''Update the parameters with the nesterov momentum
        '''
        self.ppt = self.pt

        lr = self._get_lr()
        lmbd = self.pb.lmbd
        ak, ak1 = self.theta
        self.yn = self.pt+ak*(1/ak1-1)*self.p_grad
        grad = self.pb.grad(self.yn)

        self.pt = self.pb.prox(self.yn-lr*grad, lmbd*lr)
        log.debug('Grad: {}'.format(np.max(grad)))
        log.debug('lr: {}'.format(lr))

        # Restart if needed
        lc = self.pb.cost(self.pt)
        if self.restart and lc > self.cost[-1]:
            log.debug('Restart')
            self.yn = self.ppt
            grad = self.pb.grad(self.yn)
            self.pt = self.pb.prox(self.yn-lr*grad, lmbd*lr)

        #Update momentum information
        self.p_grad = self.pt-self.ppt
        self.theta = [self._theta(), ak]

    def _theta(self):
        if self.f_theta == 'k2':
            return 2/(self.t+3)
        elif type(self.f_theta) == float:
            return 1-self.f_theta
        return 0.2
