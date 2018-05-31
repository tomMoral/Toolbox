import numpy as np


from . import _GradientDescent
from toolbox.logger import Logger

log = Logger(name='ProximalDescent')


class ProximalDescent(_GradientDescent):
    """Gradient Descent with the nesterov momentum"""
    def __init__(self, problem, decreasing_rate='', f_theta='boyd',
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
        self.p_grad = np.zeros(self.pb.pt.shape)
        self.f_theta = f_theta
        self.alpha = 1/self.pb.L

    def __repr__(self):
        return ('Proximal Descent' +
                (' with Restart' if self.restart else ''))

    def p_update(self):
        '''Update the parameters with the nesterov momentum
        '''
        self.ppt = self.pb.pt

        lr = self._get_lr()
        lmbd = self.pb.lmbd
        ak, ak1 = self.theta
        self.yn = self.pb.pt+ak*(1/ak1-1)*self.p_grad
        grad = self.pb.grad(self.yn)

        self.pb._update(self.pb.prox(self.yn-lr*grad, lmbd*lr))

        # Restart if needed
        res_cond = (self.yn - self.pb.pt).dot((self.pb.pt - self.ppt).T).sum()
        if self.restart and res_cond > 0:
            log.debug('Restart')
            self.yn = self.ppt
            grad = self.pb.grad(self.yn)
            self.pb._update(self.pb.prox(self.yn-lr*grad, lmbd*lr))

        #Update momentum information
        self.p_grad = self.pb.pt-self.ppt
        self.theta = [self._theta(ak), ak]
        return np.sum(self.p_grad*self.p_grad)

    def _theta(self, ak):
        if self.f_theta == 'boyd':
            ak2 = ak*ak
            return (np.sqrt(ak2*ak2+4*ak2)-ak2)/2
        if self.f_theta == 'k2':
            return 2/(self.t+3)
        elif type(self.f_theta) == float:
            return 1-self.f_theta
        return 0.2
