from math import sqrt


class _GradientDescent(object):
    """Class to hold gradient descent properties"""
    def __init__(self, params, alpha, grad=None, decreasing_rate='sqrt'):
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
        self.alpha = alpha
        self.params = params
        self.decreasing_rate = decreasing_rate
        self._grad = grad
        self.t = 1

    def update(self, grad=None):
        '''Update the parameters given

        Parameters
        ----------
        grad: list, optional (default: None)
            list of the gradient for each parameters
        '''
        self.t += 1
        grad = self._get_grad(grad)
        self.params = [p-self._get_lr()*dp
                       for p, dp in zip(self.params, grad)]
        return self.z

    def _get_grad(self, grad):
        '''Auxillary funciton, return the gradient
        '''
        if grad is not None:
            return grad
        elif self._grad is not None:
            return self._grad(self.params)
        raise GD_Exception('No gradient furnished!')

    def _get_lr(self):
        '''Auxillary funciton, return the learning rate
        '''
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
