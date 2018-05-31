from sys import stdout as out
import numpy as np
import logging
from multiprocessing import Process, Manager, Queue
from queue import Empty
from collections import defaultdict
from time import time

import matplotlib as mpl
mpl.interactive(True)


from . import STOP, PROGRESS, SAVE, COST, LOG, PASS, OBJ, LEVEL
from . import DEBUG


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
        PROGRESS: name/i_max/iteration
        SAVE: Object to save/optional nameFile
    """
    def __init__(self, levl=logging.DEBUG, name='root', process=None,
                 graph_update=0.4, default_line_style='-o'):
        super(Handler, self).__init__(name='Log_process')
        self.daemon = True
        self.process = process
        # Get root logger
        self.log = logging.getLogger(name)
        # Add a default handler to print in console
        if len(self.log.handlers) < 1:
            ch = logging.StreamHandler(out)
            formatter = logging.Formatter('\r%(levelname)s '
                                          '- %(message)s')
            ch.setFormatter(formatter)
            self.log.addHandler(ch)

        # Start the concurent queue
        self.qin = Queue()

        # Progress logging fields
        self.last_writter = ''
        self.unfinished = False

        # Object saving handles
        self.manager = Manager()
        self.log_objects = self.manager.dict()

        # Graphical logging handling
        self.graph = defaultdict(lambda: defaultdict(lambda: None))
        self.lst_time = defaultdict(lambda: 0)
        self.default_line_style = default_line_style
        self.graph_update = graph_update

        # Set the level  to its default value
        self.level = 0
        self.set_mode(levl)

    def get_pin(self):
        '''Return the logging queue
        '''
        return self.qin

    def set_mode(self, levl=logging.DEBUG):
        '''Change the logging level of this handler
        '''
        if DEBUG:
            self._log(10, 'HANDLER - Set mode: {}'
                      ''.format(levl))
        self.log.setLevel(levl)
        for ch in self.log.handlers:
            ch.setLevel(levl)
        self.level = levl

    def run(self):
        '''Handler loop'''
        if DEBUG or self.level < 10:
            self._log(10, 'HANDLER - Start properly with level {}'
                      ''.format(self.level))
        while True:
            try:
                action, levl, entry, logger = self._treat()
                if action == STOP:
                    print('Got stop')
                    break
                elif action == PASS:
                    continue
                elif action == LEVEL:
                    self.set_mode(levl=levl)
                if levl < self.level:
                    continue
                if action == PROGRESS:
                    self._progress(levl, **entry)
                elif action == SAVE:
                    self._save(levl, **entry)
                elif action == COST:
                    try:
                        self._graph_cost(**entry)
                    except:
                        import traceback
                        msg = traceback.format_exc()
                        self._log(40, 'HANDLER - fail to log graph cost:\n{}'
                                  '\n\n'.format(msg))
                elif action == LOG:
                    self._log(levl, entry)
                elif action == OBJ:
                    self._log_obj(**entry)
            except ValueError:
                import traceback
                msg = traceback.format_exc()
                self._log(40, "HANDLER - Fail to log {}, {}, {}"
                          "".format(action, levl, entry))
            except KeyboardInterrupt:
                pass
            except:
                import traceback
                msg = traceback.format_exc()
                msg_form = 'HANDLER\n'
                msg_form += '='*79+'\n'
                msg_form += msg+'\n'+'='*79+'\n'
                self._log(40, msg_form)
        if DEBUG or self.level < 10:
            self._log(10, 'HANDELER - Clean quit')
        return 0

    def _treat(self):
        try:
            entry = self.qin.get(True, 2)
            if DEBUG or self.level < logging.DEBUG:
                if entry[0] != OBJ:
                    self._log(10, 'HANDLER - Got entry {}'
                                  ''.format(entry))
            return entry
        except Empty:
            return PASS, None, None, None

    def _begin_line(self):
        if self.unfinished:
            out.write('\n')

    def _log(self, levl, msg, **kwargs):
        if self.unfinished:
            out.write('\n')
        self.unfinished = False
        self.last_writter = ''
        self.log.log(levl, msg, **kwargs)

    def _progress(self, levl=logging.INFO, iteration=0,
                  name='Progress', i_max=100):
        '''Function to log progress'''
        # If the previous line wasn't a current progress line
        # Start a new progress logging line
        if self.last_writter != name:
            self.lst_time[name] = 0
            self._begin_line()
            self.unfinished = True
            out.write('{} - {} - '.format(logging.getLevelName(levl), name))
            out.write(' '*7)

            #Update progresse entry
        if time() - self.lst_time[name] >= 0.1:
            self.lst_time[name] = time()
            out.write('\b'*7 + '{:7.2%}'.format(iteration/i_max))

        # End the current progress entry if the i_max is reached
        if iteration >= i_max and self.unfinished:
            out.write('\b'*7 + 'Done   \n')
            self.unfinished = False
            self.last_writter = ''

        out.flush()
        self.last_writter = name

    def _graph_cost(self, cost=1, iteration=None,
                    name='Cost', curve='cost', end=False, linestyle=None,
                    scale='log', **kwargs):
        import matplotlib.pyplot as plt
        if end:
            if DEBUG or self.level < 10:
                self._log(10, 'HANDLER - End graphical cost follow up')
            self._update_fig(name)
            line = self.graph[name]['old'+curve]
            if line is not None:
                line.remove()
            self.graph[name]['old'+curve] = self.graph[name][curve]
            self.graph[name][curve] = None
            return
        # Update the graph
        line = self.graph[name][curve]

        if line is None:
            if iteration is None:
                iteration = 1
            # Select the
            plt.figure(name)
            if scale == 'log':
                line = plt.loglog([iteration], [cost],
                                  self.default_line_style,
                                  label=curve, **kwargs)[0]
            else:
                line = plt.plot([iteration], [cost],
                                self.default_line_style,
                                label=curve, **kwargs)[0]
            plt.legend()
        else:
            x = line.get_xdata()
            if iteration is None:
                iteration = x[-1]+1
            x = np.r_[x, iteration]
            i0 = x.argsort()
            line.set_xdata(x[i0])
            line.set_ydata(np.r_[line.get_ydata(), [cost]][i0])

        if linestyle is not None:
            line.set_linestyle(linestyle)
            line.set_marker('')

        self.graph[name][curve] = line

        if time()-self.lst_time[name] >= self.graph_update:
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

    def _save(self, levl, obj, fname='.pkl'):

        if fname[0] == '.':
            fname = obj.__name__ + fname
        pass

    def _log_obj(self, name, obj, fun=None,
                 graph_cost=None, iteration=None,
                 time=None):
        if fun is None:
            fun = np.copy
        fobj = fun(obj)
        l = self.log_objects.get(name, [])
        l += [fobj]
        self.log_objects[name] = l
        if iteration is not None:
            l = self.log_objects.get(name+'_i', [])
            l += [iteration]
            self.log_objects[name+'_i'] = l
        if time is not None:
            l = self.log_objects.get(name+'_t', [])
            l += [time]
            self.log_objects[name+'_t'] = l
        if graph_cost is not None:
            graph_cost.update(iteration=iteration)
            self._graph_cost(cost=fobj, **graph_cost)
