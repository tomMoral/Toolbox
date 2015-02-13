from sys import stdout as out

import logging


class Logger(object):
    """Logging handler"""
    def __init__(self, levl=logging.INFO, **kwargs):
        super(Logger, self).__init__()

        root = logging.getLogger()
        root.setLevel(levl)
        if len(root.handlers) < 1:
            ch = logging.StreamHandler(out)
            ch.setLevel(levl)
            formatter = logging.Formatter('\r%(levelname)s '
                                          '- %(message)s')
            ch.setFormatter(formatter)
            root.addHandler(ch)

    def progress(self, i, max_iter):
        out.write('\b'*6 + '{:6.2%}'.format(i/max_iter))
        out.flush


if __name__ == '__main__':
    log = Logger(levl=logging.DEBUG)

    max_iter = 1000000

    for i in range(max_iter):
        log.progress(i, max_iter)
