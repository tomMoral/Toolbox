from toolboxTom.logger import Logger


def test_graphical_logging():
    log = Logger(levl=10)
    i_max = 1000

    for i in range(i_max):
        log.graphical_cost(cost=1/(i+1), n_iteration=i)
        if i == 50000:
            raise KeyboardInterrupt()
    log.process_queue()
    log.end()


def test_progress():
    log = Logger(levl=10)
    i_max = 10
    j_max = 100
    for i in range(i_max):
        log.progress(name='progress1', iteration=i, i_max=i_max)
        for j in range(j_max):
            log.progress(name='progress2', iteration=j, i_max=j_max)

    log.process_queue()
    log.end()


def test_level():
    log = Logger(levl=40)
    log.debug('Do not appear!!')
    log.info('Do not appear!!')
    log.warning('Do not appear!!')
    log.error('Should appear!!')
    log.critical('Should appear!!')
    log.process_queue()
    log.end()

