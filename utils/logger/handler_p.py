from sys import stdout as out
import logging
from multiprocessing import Process, Pipe
from collections import defaultdict


from . import STOP, LOG, PROGRESS, SAVE, COST


class Handler(Process):
    """Asynchronous logging handler

    Usage
    -----
    Insert logging entry in h.get_pin() with shape (a, l, kwargs) where
    a: action in {STOP, LOG, PROGRESS, SAVE}
    l: logging level
    kwargs: parameters for the action
        STOP: None
        LOG: message to log
        PROGRESS: name/max_iter/iteration
        SAVE: Object to save/optional nameFile
    """
    def __init__(self, levl=logging.INFO, name='root', **kwargs):
        super(Handler, self).__init__()
        # Get root logger
        self.log = logging.getLogger(name)
        # Add a default handler to print in console
        if len(self.log.handlers) < 1:
            ch = logging.StreamHandler(out)
            formatter = logging.Formatter('\r%(levelname)s '
                                          '- %(message)s')
            ch.setFormatter(formatter)
            self.log.addHandler(ch)
        self.last_writter = ''
        self.unfinished = False
        self.pin, self.pout = Pipe()
        self.set_mode(levl)
        self.graph = defaultdict(lambda: [[], []])

    def get_pin(self):
        '''Return the logging pipe
        '''
        return self.pin

    def set_mode(self, levl=logging.INFO):
        '''Change the logging level of this handler
        '''
        self.log.debug('Set mode: {}'.format(logging.getLevelName(levl)))
        self.log.setLevel(levl)
        for ch in self.log.handlers:
            ch.setLevel(levl)
        self.level = levl

    def run(self):
        '''Handler loop'''
        while True:
            action, levl, entry = self.pout.recv()
            if action == STOP:
                self._log(10, 'End the logger')
                return 0
            if levl < self.level:
                continue
            if action == PROGRESS:
                self._progress(levl, **entry)
            elif action == SAVE:
                self._save(levl, **entry)
            elif action == COST:
                self._graph_cost(levl, **entry)
            else:
                self._log(levl, entry)

    def _beggin_line(self):
        if self.unfinished:
            out.write('\n')

    def _log(self, levl, msg, **kwargs):
        if self.unfinished:
            out.write('\n')
        self.unfinished = False
        self.last_writter = ''
        self.log.log(levl, msg, **kwargs)

    def _progress(self, levl=logging.INFO, iteration=0,
                  name='Progress', max_iter=100):
        '''Function to log progress'''
        # If the previous line wasn't a current progress line
        # Start a new progress logging line
        if self.last_writter != name:
            self._beggin_line()
            self.unfinished = True
            out.write('{} - {} - '.format(logging.getLevelName(levl), name))
            out.write(' '*7)

        #Update progresse entry
        out.write('\b'*7 + '{:7.2%}'.format(iteration/max_iter))

        # End the current progress entry if the max_iter is reached
        if iteration >= max_iter-1:
            out.write('\b'*7 + 'Done   \n')
            self.unfinished = False

        out.flush()
        self.last_writter = name

    def _graph_cost(self, levl=logging.INFO, cost=0, iteration=1,
                    name='Cost'):

        graph = self.graph[name]
        graph[0] += [iteration]
        graph[1] += [iteration]

        import matplotlib as mpl
        mpl.interactive(True)
        import matplotlib.pyplot as plt
        plt.figure(name)
        plt.cla()
        plt.plot()

    def _save(self, levl, obj, fname='.pkl'):
        pass
