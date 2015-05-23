import numpy as np

from . import ImplementationError

naming = {'Z': 0, 'D': 1}


class _CoupledProblem(object):
    """Meta class to handle optimisation problem"""
    def __init__(self, x0=None, size=None, naming=[]):
        super(_CoupledProblem, self).__init__()
        self.x0 = x0
        if self.x0 is None:
            assert size is not None, 'No size for the Problem'
            self.x0 = [np.zeros(s) for s in size]
        self.n_params = len(self.x0)
        self.pbs = [_ParamProblem(self, i) for i in range(self.n_params)]
        self.params = [0] * self.n_params
        self.naming = {}
        for i in range(self.n_params):
            self.reset(i)
            if len(naming) > i:
                self.naming[naming[i]] = i

    def cost(self):
        raise ImplementationError('cost not implemented in', self.__class__)

    def grad(self, param, pt):
        try:
            return self.__getattribute__('grad_'+str(param))(pt)
        except AttributeError:
            raise ImplementationError('grad not implemented for param',
                                      param, 'in', self.__class__)

    def prox(self, param, pt):
        try:
            return self.__getattribute__('prox_'+str(param))(pt)
        except AttributeError:
            raise ImplementationError('prox not implemented for param',
                                      param, 'in', self.__class__)

    def soluce(self, param, dpt=None, i0=None):
        try:
            print("get", 'soluce_'+str(param))
            return self.__getattribute__('soluce_'+str(param))(dpt, i0)
        except AttributeError:
            raise ImplementationError('soluce not implemented for param {}'
                                      ''.format(param),  self.__class__)

    def _update(self, param, update):
        self.params[param] = update

    def reset(self, param, p0=None, size=None):
        if p0 is None:
            if size is None:
                p0 = self.x0[param]
            else:
                p0 = np.zeros(size)
        self._update(param, p0)

    def __getattr__(self, attr):
        if attr in naming.keys():
            i = naming[attr]
            return self.params[i]
        return super(_CoupledProblem, self
                     ).__getattribute__(attr)

    def __setattr__(self, attr, value):
        if attr in naming.keys():
            i = naming[attr]
            self.params[i] = value
        return super(_CoupledProblem, self
                     ).__setattr__(attr, value)


class _ParamProblem(object):
    def __init__(self, cpb, param):
        self.cpb = cpb
        self.param = param
        self.__initialised = True

    def __getattr__(self, attr):
        if attr == 'pt':
            return self.cpb.params[self.param]
        if attr in ['grad', 'prox', 'soluce']:
            try:
                return self.cpb.__getattr__(attr+'_'+str(self.param))

            except AttributeError:
                raise ImplementationError('{} not implemented for param {}'
                                          ''.format(attr, self.param),
                                          self.__class__)
        try:
            return super(_ParamProblem, self
                         ).__getattr__(attr)
        except AttributeError:
            return self.cpb.__getattr__(attr)

    def __setattr__(self, attr, value):

        if '_ParamProblem__initialised' not in self.__dict__.keys():
            return dict.__setattr__(self, attr, value)
        if attr == 'pt':
            self.cpb.params[self.param] = value
        elif attr == ['cpb', 'param']:
            super(_ParamProblem, self
                  ).__setattribute__(attr, value)
        else:
            self.cpb.__setattr__(attr, value)
