import numpy as np


from . import _GradientDescent
from utils.logger import Logger

log = Logger(name='ProximalDescent')


class ProximalDescent(_GradientDescent):
    """Gradient Descent with the nesterov momentum"""
    def __init__(self, problem, decreasing_rate='', f_theta='k2',
                 restart=False, **kwargs):
        super(ProximalDescent, self).__init__(
            problem, decreasing_rate, **kwargs)
        self.theta = [1, 1]
        if type(f_theta) is float:
            self.theta = [1-f_theta]*2
        self.p_grad = [np.zeros(s) for s in problem.sizes]
        self.restart = restart
        self.f_theta = f_theta
        self.alpha = 1e-4
        log.debug('Test debug')

    def __repr__(self):
        return ('Proximal Descent'  +
                (' with Restart' if self.restart else ''))

    def p_update(self):
        '''Update the parameters with the nesterov momentum
        '''
        self.ppt = self.pt
        lr = self._get_lr()
        ak, ak1 = self.theta
        self.yn = [p+ak*(1/ak1-1)*pg for p, pg in zip(self.pt, self.p_grad)]
        grad = self.pb.grad(self.yn)

        self.pt = [self.pb.prox(y-ll*dy)
                   for ll, y, dy in zip(lr, self.yn, grad)]

        # Restart if needed
        lc = self.pb.cost(self.pt)
        if self.restart and lc > self.cost[-1]:
            log.debug('Restart')
            self.yn = self.ppt
            grad = self.pb.grad(self.yn)
            self.pt = [self.pb.prox(y-lr*dy) for y, dy in zip(self.yn, grad)]

        #Update momentum information
        self.p_grad = [p-pp for p, pp in zip(self.pt, self.ppt)]
        self.theta = [self._theta(), ak]

    def _theta(self):
        if self.f_theta == 'k2':
            return 2/(self.t+3)
        elif type(self.f_theta) == float:
            return 1-self.f_theta
        return 0.2
