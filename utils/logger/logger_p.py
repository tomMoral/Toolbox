import logging


from . import STOP, LOG, PROGRESS, SAVE, MODE, COST
from .handler_p import Handler


class Logger(object):
    """Asynchronous Logger

    Usage
    -----
    use debug/info/warning/error/critical to log entries
    use progress to handle progress in loop
    use end to stop the logger
    """
    console = Handler(levl=20)
    console.start()
    references = 0
    _alive = True

    def restart():
        if not Logger.console.is_alive():
            print('new console')
            Logger.console = Handler(levl=20)
            Logger.console.start()
            Logger._alive = True
        return Logger.console.get_pin()

    def __init__(self, name='', levl=None):
        '''Create a basic Logger and add a console handler
        '''
        super(Logger, self).__init__()
        self.level = logging.DEBUG
        Logger.references += 1
        self.qin = Logger.restart()
        if levl is not None:
            self.set_mode(levl)

    def _log(self, entry):
        try:
            self.qin.put(entry, False)
        except TimeoutError:
            print('Fail to log', entry)

    def set_mode(self, levl=logging.INFO):
        '''Change the logging level of the logger and the console handler
        '''
        #self.qin.send((MODE, self.level, levl))
        self._log((MODE, 100, levl))
        self.level = levl

    def end(self):
        '''Finish to handle the log entry and stop the console handler
        '''
        Logger.references -= 1
        if Logger.references == 0 and Logger._alive:
            self._log((STOP, None, None))
            Logger.console.join()
            Logger._alive = False

    def kill(self):
        if Logger._alive:
            self._log((LOG, 10, 'kill'))
            self._log((STOP, None, None))
            Logger.console.join()
            Logger._alive = False

    def is_alive(self):
        return self._alive

    def debug(self, msg, **kwargs):
        if self.level <= 10:
            self._log((LOG, 10,  msg))

    def info(self, msg, **kwargs):
        if self.level <= 20:
            self._log((LOG, 20, msg))

    def warning(self, msg, **kwargs):
        if self.level <= 30:
            self._log((LOG, 30, msg))

    def error(self, msg, **kwargs):
        if self.level <= 40:
            self._log((LOG, 40, msg))

    def critical(self, msg, **kwargs):
        if self.level <= 50:
            self._log((LOG, 50, msg))

    def progress(self, iteration=0, max_iter=100, name='Progress',
                 levl=logging.INFO, **kwargs):
        '''Log a progression, typically for a loop,

        Parameters
        ----------
        iteration: Avancement of the loop
        max_iter: Total # of iteration
        name: Name of the loop, to diferentiate the loops
        levl: Level of the log
        '''
        if self.level <= levl:
            kwargs.update(dict(iteration=iteration, max_iter=max_iter,
                               name=name))
            self._log((PROGRESS, levl, kwargs))

    def graphical_cost(self, cost=0, iteration=1, name='Cost',
                       levl=logging.INFO, end=False, **kwargs):
        '''Log a progression, typically for a loop,

        Parameters
        ----------
        cost: cost to log
        n_iteration: #iter depuis le dernier log
        name: Name of the loop, to diferentiate the loops
        levl: Level of the log
        '''
        if self.level <= levl:
            kwargs.update(dict(cost=cost, iteration=iteration,
                               name=name, end=end))
            self._log((COST, levl, kwargs))

    def save(self, levl=logging.INFO, **kwargs):
        if self.level <= levl:
            self._log((SAVE, levl, kwargs))


if __name__ == '__main__':
    log = Logger()
    max_iter = 1000000

    for i in range(max_iter):
        log.graphical_cost(cost=1/(i+1), n_iteration=1)

    log.end()
