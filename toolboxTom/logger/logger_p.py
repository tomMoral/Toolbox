import logging
import multiprocessing


from . import STOP, LOG, PROGRESS, SAVE, COST, OBJ, LEVEL
from .handler_p import Handler

from . import DEBUG


class Logger(object):
    """Asynchronous Logger

    Usage
    -----
    use debug/info/warning/error/critical to log entries
    use progress to handle progress in loop
    use end to stop the logger
    """
    output = None
    references = 0
    _alive = False

    @staticmethod
    def restart():
        if Logger.output is None or not Logger.output.is_alive():
            process = multiprocessing.current_process()
            Logger.output = Handler(levl=10, process=process)
            Logger.output.start()
            Logger._alive = True
        return Logger.output.get_pin()

    def __init__(self, name='', levl=logging.INFO):
        '''Create a basic Logger and add a output handler
        '''
        super(Logger, self).__init__()
        Logger.references += 1
        self.name = name
        self.running = True
        self.set_level(levl)

    def __del__(self):
        if self.running:
            self.end()

    def _log(self, entry):
        try:
            qin = Logger.restart()
            assert type(entry) == tuple
            assert len(entry) == 3
            qin.put(entry+(self.name,), False)
        except TimeoutError:
            print('== ERROR - LOGGER - Fail to log', entry)
        except AssertionError:
            print('== ERROR - LOGGER - Bad entry in logger {}'
                  '\n'.format(self.name))

    def _format(self, msg, *args):
        msg = str(msg)
        if len(args) > 0:
            msg += ' - ' + ' - '.join(
                [str(a) for a in args])
        if self.name != '':
            return self.name + ' - ' + msg
        return msg

    def set_level(self, levl=logging.INFO):
        '''Change the logging level of the logger and the output handler
        '''
        self.level = levl
        if self.level <= 10:
            self._log((LEVEL, self.level, None))
        if DEBUG:
            self.debug('Set level to {}'.format(levl))

    def end(self):
        '''Finish to handle the log entry and stop the output handler
        '''
        Logger.references -= 1
        if (Logger.references == 0 and Logger._alive and
                Logger.output.is_alive()):
            self._log((STOP, None, None))
            Logger.output.join()
            Logger._alive = False
        self.running = False

    def kill(self):
        if Logger._alive:
            self._log((LOG, 10, 'kill'))
            self._log((STOP, None, None))
            Logger.output.join()
            Logger._alive = False
        self.running = False

    def is_alive(self):
        return self._alive

    def debug(self, msg, *args):
        if self.level <= 10:
            self._log((LOG, 10,  self._format(msg, *args)))

    def info(self, msg, *args):
        if self.level <= 20:
            self._log((LOG, 20, self._format(msg, *args)))

    def warning(self, msg, *args):
        if self.level <= 30:
            self._log((LOG, 30, self._format(msg, *args)))

    def error(self, msg, *args):
        if self.level <= 40:
            self._log((LOG, 40, self._format(msg, *args)))

    def critical(self, msg, *args):
        if self.level <= 50:
            self._log((LOG, 50, self._format(msg, *args)))

    def progress(self, iteration=0, i_max=100, name='Progress',
                 levl=logging.INFO, *args, **kwargs):
        '''Log a progression, typically for a loop,

        Parameters
        ----------
        iteration: Avancement of the loop
        i_max: Total # of iteration
        name: Name of the loop, to diferentiate the loops
        levl: Level of the log
        '''
        if self.name != '':
            name = self.name + ' - ' + name
        if self.level <= levl:
            kwargs.update(dict(iteration=iteration, i_max=i_max,
                               name=name))
            self._log((PROGRESS, levl, kwargs))

    def log_obj(self, levl=20, name='', obj=None, **kwargs):
        if self.level <= levl and name != '' and obj is not None:
            kwargs.update(dict(name=name, obj=obj))
            self._log((OBJ, levl, kwargs))

    def graphical_cost(self, cost=0, iteration=None, name='Cost',
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

    def process_queue(self):
        import time
        while Logger.output.qin.qsize() != 0:
            time.sleep(0.4)
