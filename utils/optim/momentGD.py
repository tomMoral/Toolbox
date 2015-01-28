import numpy as np

from . import _GradientDescent, GD_Exception


class MomentGradientDescent(_GradientDescent):
    """Gradient Descent with moment update"""
    def __init__(self, params, alpha, alpha_moment=0.4, grad=None,
                 decreasing_rate='sqrt'):
        super(MomentGradientDescent, self).__init__(
            params, alpha, grad, decreasing_rate)
        self.alpha_moment = alpha_moment
        self.p_grad = [np.zeros(np.shape(p)) for p in params]

    def update(self, grad=None):
        '''Update the parameters with a moment

        Parameters
        ----------
        grad: list, optional (default: None)
            list of the gradient for each parameters
        '''
        self.t += 1
        grad = self.get_grad(grad)
        self.p_grad = [dp + self.alpha_moment*pg
                       for dp, pg in zip(grad, self.p_grad)]
        lr = self._get_lr()
        self.params = [p - lr*dp
                       for p, dp in zip(self.params, self.p_grad)]
        return self.params


class NesterovMomentGradientDescent(_GradientDescent):
    """Gradient Descent with the nesterov momentum"""
    def __init__(self, params, alpha, alpha_moment=0.4, grad=None,
                 decreasing_rate='sqrt'):
        super(NesterovMomentGradientDescent, self).__init__(
            params, alpha, grad, decreasing_rate)
        self.alpha_moment = alpha_moment
        self.p_grad = [np.zeros(np.shape(p)) for p in params]
        if grad is not None:
            raise GD_Exception('Nesterov accelerated gradient need a'
                               ' gradient computation function for grad')

    def update(self):
        '''Update the parameters with the nesterov momentum
        '''
        self.t += 1
        pz = self.params
        lr = self._get_lr()
        self.params = [p+pg for p, pg in zip(self.params, self.p_grad)]
        grad = self._get_grad()
        self.params = [p-lr*dp for p, dp in zip(self.params, grad)]
        self.p_grad = [self.alpha_moment*(p-pp)
                       for p, pp in zip(self.params, pz)]
        return self.params
