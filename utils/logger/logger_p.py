import logging


from . import STOP, LOG, PROGRESS, SAVE
from .handler_p import Handler


class Logger(object):
    """docstring for Log"""
    def __init__(self, levl=logging.INFO):
        self.handler = Handler(levl=levl)
        self.qin = self.handler.get_pin()
        self.handler.start()
        self.level = levl

    def __del__(self):
        self.end()

    def set_mode(self, levl=logging.INFO):
        self.handler.set_mode(levl)

    def end(self):
        self.qin.send((STOP, None, None))
        self.handler.join()

    def debug(self, msg, **kwargs):
        if self.level <= 10:
            self.qin.send((LOG, 10,  msg))

    def info(self, msg, **kwargs):
        if self.level <= 20:
            self.qin.send((LOG, 20, msg))

    def warning(self, msg, **kwargs):
        if self.level <= 30:
            self.qin.send((LOG, 30, msg))

    def error(self, msg, **kwargs):
        if self.level <= 40:
            self.qin.send((LOG, 40, msg))

    def critical(self, msg, **kwargs):
        if self.level <= 50:
            self.qin.send((LOG, 50, msg))

    def progress(self, levl=logging.INFO, **kwargs):
        if self.level <= levl:
            self.qin.send((PROGRESS, levl, kwargs))

    def save(self, levl=logging.INFO, **kwargs):
        if self.level <= levl:
            self.qin.send((SAVE, levl, kwargs))


if __name__ == '__main__':
    log = Logger()

    max_iter = 1000000

    for i in range(max_iter):
        log.progress(name='Counter', i=i, max_iter=max_iter)

    log.end()
