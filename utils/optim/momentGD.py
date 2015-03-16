import numpy as np


from . import _GradientDescent


class MomentGradientDescent(_GradientDescent):
    """Gradient Descent with moment update"""
    def __init__(self, problem, decreasing_rate='', alpha_moment=0.9,
                 restart=True, **kwargs):
        super(MomentGradientDescent, self).__init__(
            problem, decreasing_rate, **kwargs)
        self.p_grad = [np.zeros(s) for s in problem.sizes]
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
        self.ppt = self.pt
        grad = self.pb.grad(self.pt)
        self.p_grad = [dp + self.alpha_moment*pg
                       for dp, pg in zip(grad, self.p_grad)]
        lr = self._get_lr()
        self.pt = [p-l*dp for l, p, dp in zip(lr, self.pt, self.p_grad)]
        if self.restart and self.pb.cost(self.pt) > self.cost[-1]:
            self.pt = self.ppt
            grad = self.pb.grad(self.pt)
            self.p_grad = grad
            self.pt = [p-lr*dp for p, dp in zip(self.pt, grad)]


class NesterovMomentGradientDescent(_GradientDescent):
    """Gradient Descent with the nesterov momentum"""
    def __init__(self, problem, decreasing_rate='', alpha_moment=0.9,
                 restart=False, **kwargs):
        super(NesterovMomentGradientDescent, self).__init__(
            problem, decreasing_rate, **kwargs)
        self.alpha_moment = alpha_moment
        self.p_grad = [np.zeros(s) for s in problem.sizes]
        self.restart = restart
        '''if grad is not None:
            raise GD_Exception('Nesterov accelerated gradient need a'
                               ' gradient computation function for grad')'''

    def __repr__(self):
        return 'NesterovMomentDescent - '+str(self.restart)

    def p_update(self):
        '''Update the parameters with the nesterov momentum
        '''
        self.ppt = self.pt
        lr = self._get_lr()

        # Compute the intermediate step
        self.yn = [p+pg for p, pg in zip(self.pt, self.p_grad)]

        # Update
        grad = self.pb.grad(self.yn)
        self.pt = [p-lr*dp for p, dp in zip(self.yn, grad)]

        # Restart criterion
        if self.restart and self.pb.cost(self.pt) > self.cost[-1]:
            self.pt = self.ppt
            grad = self.pb.grad(self.pt)
            self.pt = [p-lr*dp for p, dp in zip(self.pt, grad)]

        # Save movement direction
        self.p_grad = [self.alpha_moment*(p-pp)
                       for p, pp in zip(self.pt, self.ppt)]
