import numpy as np

from optim.problem import _Problem
from utils.logger import Logger
log = Logger()


class Test(_Problem):
    """Test class for problem"""
    def __init__(self, lr=.1, **kwargs):
        super(Test, self).__init__(sizes=(1000, ))
        A = np.random.random(size=(100, 1000))
        A = A < 0.07
        A = A*np.random.normal(scale=2, size=A.shape)
        self.A = A + A[::-1]
        self.b = np.random.normal(size=(100, ))
        self.lr = lr
        self.L = np.sum(A*A)

    def cost(self, pt):
        res = self.A.dot(pt[0]) - self.b
        res = np.sum(res*res)
        return 0.5*res + self.lr*np.sum(abs(pt[0]))

    def grad(self, pt):
        return [self.A.T.dot(self.A.dot(pt[0]) - self.b)]

    def prox(self, pt):
        return np.sign(pt)*np.maximum(abs(pt)-self.lr, 0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Optimization and logging tests')
    parser.add_argument('--log_cost', action='store_true',
                        help='Graphical cost log')

    args = parser.parse_args()

    try:
        pb = Test(lr=0)

        from optim import _GradientDescent
        from optim.momentGD import (MomentGradientDescent,
                                    NesterovMomentGradientDescent)
        from optim.proximal import ProximalDescent
        gd = _GradientDescent(pb, decreasing_rate='')
        gdm = MomentGradientDescent(pb, decreasing_rate='',
                                    alpha_moment=0.8, restart=True)
        gdnm = NesterovMomentGradientDescent(pb, decreasing_rate='',
                                             alpha_moment=0.8, restart=True)
        pd = ProximalDescent(pb, restart=False)
        pdr = ProximalDescent(pb, restart=True)
        pdr2 = ProximalDescent(pb, restart=True, f_theta=0.8)
        i, go = 0, [True, True, True]
        while i < 100 and np.any(go):
            go = [gd.update(),
                  gdm.update(),
                  gdnm.update(),
                  pd.update(),
                  pdr.update(),
                  pdr2.update()]
            i += 1
            log.progress(name='Gradient Optimisation', iteration=i,
                         i_max=100)
            if args.log_cost:
                log.graphical_cost(name='Test', cost=gd.cost[-1], iteration=i,
                                   curve='GD')
                log.graphical_cost(name='Test', cost=gdm.cost[-1], iteration=i,
                                   curve='Moementum')
                log.graphical_cost(name='Test', cost=gdnm.cost[-1], iteration=i,
                                   curve='Nesterov Momentum')
                log.graphical_cost(name='Test', cost=pd.cost[-1], iteration=i,
                                   curve='Proximal Descent')
                log.graphical_cost(name='Test', cost=pdr.cost[-1], iteration=i,
                                   curve='PDR')
                log.graphical_cost(name='Test', cost=pdr2.cost[-1], iteration=i,
                                   curve='PDR2')
        if args.log_cost:
            log.graphical_cost(name='Test', end=True)

        import matplotlib.pyplot as plt
        plt.loglog(gd.cost, label='GD')
        plt.loglog(gdm.cost, label='Momentum')
        plt.loglog(gdnm.cost, label='Nesterov Momentum')
        plt.loglog(pd.cost, label='Proximal')
        plt.loglog(pdr.cost, label='Proximal restart')
        plt.loglog(pdr2.cost, label='Proximal restart, fix theta')
        plt.legend()
        print('\nGap: {:6.2%}'
              ''.format((min(pdr.cost)-min(pdr2.cost))/min(pdr.cost)))
        print('Gap: {:6.2%}'
              ''.format((min(pdr.cost)-min(gdm.cost))/min(pdr.cost)))
        plt.show()
    finally:
        log.kill()
