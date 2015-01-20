from math import sqrt


class GradientDescent(object):
    """Class to hold gradient descent properties"""
    def __init__(self, z0, alpha, grad=None, decreasing_rate='sqrt'):
        self.alpha = alpha
        self.z = z0
        self.decreasing_rate = decreasing_rate
        self._grad = grad
        self.t = 1

    def update(self, grad=None):
        self.t += 1
        self.z -= self._get_lr()*self.get_grad(grad)
        return self.z

    def get_grad(self, grad):
        if grad is not None:
            return grad
        elif self.grad is not None:
            return self.grad(self.z)
        raise GD_Exception('No gradient furnished!')

    def _get_lr(self):
        lr = self.alpha
        if self.decreasing_rate == 'sqrt':
            lr /= sqrt(self.t)
        elif self.decreasing_rate == 'linear':
            lr /= self.t
        return lr


class GD_Exception(Exception):
    """Gradient Descent exception"""
    def __init__(self, msg):
        super(GD_Exception, self).__init__()
        self.msg = msg
