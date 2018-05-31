import numpy as np
import sys
from toolboxTom.logger import Logger
import multiprocessing
from multiprocessing import Process, Queue
log = Logger(name='map')
log.set_level(0)


class WorkerGroups(Process):
    def __init__(self, qin, qout, fun, id_w=0, **kwargs):
        super(WorkerGroups, self).__init__(name='Worker n°{}'.format(id_w))
        self.qin = qin
        self.qout = qout
        self.id = id_w
        self.fun = fun
        self.args = kwargs

    def run(self):
        idp, p = self.qin.get()
        while idp is not None:
            log.debug('Worker {} - |qin| = {}'
                      ''.format(self.id, self.qin.qsize()))
            params = dict(p=p, **self.args)
            try:
                self.qout.put((idp, self.fun(**params)))
            except:
                import traceback
                msg = traceback.format_exc()
                print(msg)
            idp, p = self.qin.get()
        log.debug('Worker {} finished'.format(self.id))
        return 0


def map_grouping(fun, l1, njobs=0, **kwargs):

    if njobs < 1:
        cc = multiprocessing.cpu_count()
        njobs = min(cc-1, cc+njobs)

    # Init loop variables
    N = len(l1)
    qin = Queue()
    qout = Queue()
    resultat = []

    for k, p in enumerate(l1):
        qin.put((k, p))

    try:
        slaves = []
        for i in range(njobs):
            qin.put((None, None))
            slaves += [WorkerGroups(qin, qout, fun, id_w=i, **kwargs)]
            slaves[-1].start()
        print(qin.qsize())

        idp, r = qout.get(True, 10)
        resultat = np.empty(N, dtype=type(r))
        resultat[idp] = r
        for i in range(N-1):
            idp, r = qout.get(True, 5)
            resultat[idp] = r
            log.progress(i_max=N, iteration=i,
                         name='Scoring gouping')
        for s in slaves:
            s.join()
    except:
        import traceback
        msg = traceback.format_exc()
        log.error('Map - {}'.format('fail'))
        log.error('-'*79+'\n'+msg+'\n'+'-'*79)
    finally:
        for s in slaves:
            if s.is_alive():
                s.terminate()
        return resultat


from time import time, sleep


def fun(p):
    sleep(0.001)
    return p*2

if __name__ == '__main__':
    try:
        test = np.arange(1000)
        t1 = time()
        res = map_grouping(fun, test)
        print('Multi core maping: {:.2}s'.format(time()-t1))
        t1 = time()
        res2 = list(map(fun, test))
        print('Std maping: {:.2}s'.format(time()-t1))

        assert((res == res2).all())
    finally:
        log.end()
