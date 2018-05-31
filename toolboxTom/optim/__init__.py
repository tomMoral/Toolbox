

def get_log_rate(lr):
    if lr[:3] == 'log':
        inc = float(lr[3:]) if len(lr[3:]) > 0 else 2
        log_rate = (lambda s: lambda t: s*t + (t == 0))(inc)
    elif lr[:3] == 'lin':
        inc = int(lr[3:]) if len(lr[3:]) > 0 else 10
        log_rate = (lambda s: lambda t: t+s)(inc)
    elif lr == 'none':
        log_rate = lambda t: 1e300
    elif type(lr) is int or type(lr) is float:
        inc = log_rate
        log_rate = (lambda s: lambda t: s*t + (t == 0))(inc)
    else:
        assert False, "{} is not a log rate".format(lr)

    return log_rate


class ImplementationError(Exception):
    """Implementation Error"""
    def __init__(self, msg, cls):
        super(ImplementationError, self).__init__()
        self.msg = msg
        self.cls = cls

    def __repr__(self):
        return self.cls+'-'+self.msg


from ._gradient_descent import _GradientDescent
