from sys import stdout as out
import numpy as np
import logging
from multiprocessing import Process, Pipe, Manager, Lock
from collections import defaultdict
from time import time
from collections import defaultdict


from . import STOP, PROGRESS, SAVE, COST, MODE

manager = Manager()


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
    def __init__(self, levl=logging.INFO, name='root', graph_update=0.005,
                 **kwargs):
        super(Handler, self).__init__(name='Log_process')
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
        self.level = manager.Value('i', 0)
        self.lock = Lock()
        self.set_mode(levl)
        self.graph = defaultdict(lambda: defaultdict(lambda: None))
        self.lst_time = defaultdict(lambda: 0)

    def get_pin(self):
        '''Return the logging pipe
        '''
        return self.pin

    def set_mode(self, levl=logging.INFO):
        '''Change the logging level of this handler
        '''
        self._log(logging.DEBUG, 'LOGGER - Set mode: {}'
                  ''.format(logging.getLevelName(levl)))
        self.log.setLevel(levl)
        for ch in self.log.handlers:
            ch.setLevel(levl)
        self.level = levl

    def run(self):
        '''Handler loop'''
        try:
            while True:
                action, levl, entry = self.pout.recv()
                if action == STOP:
                    self._log(logging.DEBUG, 'LOGGER - End the logger')
                    return 0
                if levl < self.level:
                    continue
                if action == PROGRESS:
                    self._progress(levl, **entry)
                elif action == SAVE:
                    self._save(levl, **entry)
                elif action == COST:
                    self._graph_cost(levl, **entry)
                elif action == MODE:
                    self.set_mode(entry)
                else:
                    self._log(levl, entry)
        except KeyboardInterrupt:
            if 10 >= self.level:
                self._log(10, 'LOGGER - KeyboardInterrupt')
        except:
            import sys
            e, v = sys.exc_info()[:2]
            print('Msg', v)
            print("Test: %s - %s" % e.__name__, v)
        finally:
            print('HANDELER - quit')
            return 0

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
            self.lst_time[name] = 0
            self._beggin_line()
            self.unfinished = True
            out.write('{} - {} - '.format(logging.getLevelName(levl), name))
            out.write(' '*7)

            #Update progresse entry
        if time() - self.lst_time[name] >= 0.1:
            self.lst_time[name] = time()
            out.write('\b'*7 + '{:7.2%}'.format(iteration/max_iter))

        # End the current progress entry if the max_iter is reached
        if iteration >= max_iter-1:
            out.write('\b'*7 + 'Done   \n')
            self.unfinished = False
            self.last_writter = ''

        out.flush()
        self.last_writter = name

    def _graph_cost(self, levl=logging.INFO, cost=0, iteration=1,
                    name='Cost', curve='cost', end=False):
        import matplotlib as mpl
        mpl.interactive(True)
        import matplotlib.pyplot as plt
        if end:
            print('end')
            self._update_fig(name)
            return

        # Update the graph
        line = self.graph[name][curve]
        if line is None:
            plt.figure(name)
            line = plt.loglog([iteration+1], [cost], '-o', label=curve)[0]
            plt.legend()
        else:
            line.set_xdata(np.r_[line.get_xdata(), iteration])
            line.set_ydata(np.r_[line.get_ydata(), cost])
        self.graph[name][curve] = line

        if time()-self.lst_time[name] >= 0.1:
            self._update_fig(name)

    def _update_fig(self, name):
        import matplotlib.pyplot as plt
        fig = plt.figure(name)
        plt.draw()
        ax = fig.get_axes()[0]
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        self.lst_time[name] = time()
        print('Updated')

    def _save(self, levl, obj, fname='.pkl'):

        if fname[0] == '.':
            fname = obj.__name__ + fname
        pass
