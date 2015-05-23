import numpy as np
from utils.logger import Logger
log = Logger('MomentGD')


from . import _GradientDescent


class MomentGradientDescent(_GradientDescent):
    """Gradient Descent with moment update"""
    def __init__(self, problem, decreasing_rate='', alpha_moment=0.9,
                 restart=True, debug=0, **kwargs):
        if debug:
            debug -= 1
            log.set_level(10)
        super(MomentGradientDescent, self).__init__(
            problem, decreasing_rate, debug=debug, **kwargs)
        self.p_grad = np.zeros(self.pb.pt.shape)
        self.alpha_moment = alpha_moment
        self.restart = restart

    def __repr__(self):
        return 'MomentDescent - '+str(self.restart)

    def p_update(self):
        '''Update the parameters with a moment

        Parameters
        ----------
        grad: list, optional (default: None)
            list of the gradient for each parameters
        '''
        self.ppt = np.copy(self.pb.pt)
        grad = self.pb.grad(self.pb.pt)
        self.p_grad = grad + self.alpha_moment*self.p_grad
        lr = self._get_lr()
        self.pt -= lr*self.p_grad
        if self.restart and self.pb.cost(self.pb.pt) > self.cost[-1]:
            self.pb._update(self.ppt)
            grad = self.pb.grad()
            self.p_grad = grad
            self.pt -= lr*grad


class NesterovMomentGradientDescent(_GradientDescent):
    """Gradient Descent with the nesterov momentum"""
    def __init__(self, problem, decreasing_rate='', alpha_moment=0.9,
                 restart=False, debug=0, **kwargs):
        if debug:
            debug -= 1
            log.set_level(10)
        super(NesterovMomentGradientDescent, self).__init__(
            problem, decreasing_rate, debug=debug, **kwargs)
        self.alpha_moment = alpha_moment
        self.p_grad = np.zeros(self.pt.shape)
        self.restart = restart

    def __repr__(self):
        return ('Nesterov Moment Descent' +
                (' with Restart' if self.restart else ''))

    def p_update(self):
        '''Update the parameters with the nesterov momentum
        '''
        self.ppt = self.pb.pt
        lr = self._get_lr()

        # Compute the intermediate step
        self.yn = self.pb.pt + self.p_grad

        # Update
        grad = self.pb.grad(self.yn)
        self.pb._update(self.yn-lr*grad)

        # Restart criterion
        if self.restart and self.pb.cost() > self.cost[-1]:
            self.pb._update(self.ppt)
            grad = self.pb.grad()
            self.pt -= lr*grad

        # Save movement direction
        self.p_grad = self.alpha_moment*(self.pb.pt-self.ppt)
