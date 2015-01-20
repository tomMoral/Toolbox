import numpy as np

from optim import *


class MomentGradientDescent(GradientDescent):
    """Gradient Descent with moment update"""
    def __init__(self, z0, alpha, alpha_moment=0.4, grad=None,
                 decreasing_rate='sqrt'):
        super(MomentGradientDescent, self).__init__(
            z0, alpha, grad, decreasing_rate)
        self.alpha_moment = alpha_moment
        self.p_grad = np.zeros(np.shape(z0))

    def update(self, grad):
        self.p_grad = grad + self.alpha_moment*self.p_grad
        self.t += 1
        self.z -= self._get_lr()*grad
        return self.z


class NesterovMomentGradientDescent(GradientDescent):
    """Gradient Descent with moment update"""
    def __init__(self, z0, alpha, alpha_moment=0.4, grad=None,
                 decreasing_rate='sqrt'):
        super(MomentGradientDescent, self).__init__(
            z0, alpha, grad, decreasing_rate)
        self.alpha_moment = alpha_moment
        self.p_grad = np.zeros(np.shape(D0))
        if grad is not None:
            raise GD_Exception('Nesterov accelerated gradient need a'
                               ' gradient computation function for grad')

    def update(self, grad=None):
        pz = np.copy(self.z)
        lr = self._get_lr()
        self.z += lr*self.p_grad
        self.z += lr*self.get_grad(grad)
        self.p_grad = self.alpha_moment*(self.z-pz)
        return self.z
