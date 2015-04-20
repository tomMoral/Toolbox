import numpy as np


class ImplementationError(Exception):
    """Implementation Error"""
    def __init__(self, msg, cls):
        super(ImplementationError, self).__init__()
        self.msg = msg
        self.cls = cls

    def __repr__(self):
        return self.cls+'-'+self.msg


class _Problem(object):
    """Meta class to handle optimisation problem"""
    def __init__(self, x0=None, size=None):
        super(_Problem, self).__init__()
        self.x0 = x0
        if self.x0 is None:
            assert size is not None, 'No size for the Problem'
            self.x0 = np.zeros(size)
        self.sizes = self.x0.shape
        self.L = 10

    def cost(self, point):
        raise ImplementationError('cost not implemented', self.__class__)

    def grad(self, point):
        raise ImplementationError('grad not implemented', self.__class__)

    def prox(self, point):
        raise ImplementationError('prox not implemented', self.__class__)

    def _update(self, pt):
        return self.prox(pt - 1/self.L*self.grad(pt)[0])


if __name__ == '__main__':
    class Test(_Problem):
        """Test class for problem"""
        def __init__(self, lr=0.001, x0=None, **kwargs):
            super(Test, self).__init__(x0, sizes=1000)
            A = np.random.random(size=(100, 1000))
            A = A < 1
            self.A = A*np.random.normal(size=A.shape)
            self.b = np.random.normal(0, 5, size=100)
            self.lr = lr
            self.L = np.sum(A*A)

        def cost(self, pt):
            res = self.A.dot(pt) - self.b
            res = res.dot(res)
            return 0.5*res + self.lr*np.sum(abs(pt))

        def grad(self, pt):
            return [self.A.T.dot(self.A.dot(pt) - self.b)]

        def prox(self, pt):
            return np.sign(pt)*np.maximum(abs(pt)-self.lr, 0)

    x0 = [np.random.normal(size=1000)]
    pb = Test(x0=x0)

    N_max = 500
    tol = 1e-29
    xp = x0
    x = x0
    theta = [1, 1]
    cost = [pb.cost(x)]
    for i in range(N_max):
        yn = [x[0] + theta[0]*(1/theta[1]-1)*(x[0]-xp[0])]
        xp = x
        x = pb._update(yn)
        theta[1] = theta[0]
        theta[0] = 2/(i+3)
        cost += [pb.cost(x)]

        # Restart?
        if cost[-1] > cost[-2]:
            yn = xp
            x = pb._update(yn)
            cost[-1] = pb.cost(x)
        print(i, cost[-1])

        if len(cost) > 5 and cost[-4] - cost[-1] < tol:
            break

    xp = x0
    x = x0
    theta = [1, 1]
    cost1 = [pb.cost(x)]
    for i in range(N_max):
        yn = [x[0] + theta[0]*(1/theta[1]-1)*(x[0]-xp[0])]
        xp = x
        x = pb._update(yn)
        theta[1] = theta[0]
        theta[0] = 2/(i+3)
        cost1 += [pb.cost(x)]

        # Restart?
        print(i, cost1[-1])

        if len(cost1) > 5 and cost1[-1] < cost1[-4] < cost1[-1] + tol:
            break

    x = x0
    cost2 = [pb.cost(x)]
    for i in range(N_max):
        x = pb._update(x)
        cost2 += [pb.cost(x)]
        print(i, cost2[-1])

        if len(cost2) > 5 and cost2[-1] < cost2[-4] < cost2[-1] + tol:
            break

    from utils.optim.momentGD import MomentGradientDescent
    from utils.optim.proximal import ProximalDescent

    gdm = MomentGradientDescent(pb, decreasing_rate='',
                                alpha_moment=0.8, restart=True)
    pdr = ProximalDescent(pb, restart=True, tol=1e-29)
    for i in range(300):
        gdm.update()
        pdr.update()
    pb.x0 = gdm.pt
    pdr2 = ProximalDescent(pb, restart=True, tol=1e-29, stop='none')
    for i in range(300):
        pdr2.update()

    import matplotlib.pyplot as plt
    plt.close()
    plt.semilogy(cost, label='algo restart')
    plt.semilogy(cost1, label='algo 1')
    plt.semilogy(cost2, label='algo classic')
    plt.semilogy(gdm.cost, label='algo moment')
    plt.semilogy(pdr.cost, label='algo prox')
    plt.semilogy(pdr2.cost, label='algo prox 2')
    plt.legend()
    plt.hlines([min(cost), min(cost1), min(cost2)], 0, 2000,
               colors=['b', 'g', 'r'], linestyles='--')
    plt.show()

    print('Gap algo2 - classic:', min(cost)-min(cost2))
    print('Gap algo1 - classic:', min(cost1)-min(cost2))
