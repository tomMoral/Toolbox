from sys import stdout as out

import logging


class Logger(logging.Logger):
    """Logging handler"""
    def __init__(self, levl=logging.INFO, name='root',
                 **kwargs):
        super(Logger, self).__init__(name, **kwargs)
        self.setLevel(levl)
        if len(self.handlers) < 1:
            ch = logging.StreamHandler(out)
            ch.setLevel(levl)
            formatter = logging.Formatter('\r%(levelname)s '
                                          '- %(message)s')
            ch.setFormatter(formatter)
            self.addHandler(ch)
        self.last_wr = ''
        self.is_prog = False

    def progress(self, name, i, i_max, levl=logging.INFO):
        if self.level > levl:
            return
        if self.last_wr != name:
            if self.is_prog:
                out.write('\n')
            self.is_prog = True
            out.write('{} - {} - '.format(logging.getLevelName(levl), name))
            out.write(' '*7)
            self.last_wr = name
        out.write('\b'*7 + '{:7.2%}'.format(i/i_max))
        if i == i_max-1:
            out.write('\r{} - {} - Done   '
                      ''.format(logging.getLevelName(levl), name))
            out.write('\n')
            self.is_prog = False
            self.last_wr = ''
        out.flush()

    def set_mode(self, levl=logging.INFO):
        self.debug('Set mode: {}'.format(logging.getLevelName(levl)))
        self.setLevel(levl)
        for ch in self.handlers:
            ch.setLevel(levl)

    def debug(self, msg, **kwargs):
        if self.level > 10:
            return
        if self.is_prog:
            out.write('\n')
        self.is_prog = False
        self.last_wr = ''
        super(Logger, self).debug(msg, **kwargs)

    def info(self, msg, **kwargs):
        if self.level > 20:
            return
        if self.is_prog:
            out.write('\n')
        self.is_prog = False
        self.last_wr = ''
        super(Logger, self).info(msg, **kwargs)

    def warning(self, msg, **kwargs):
        if self.level > 30:
            return
        if self.is_prog:
            out.write('\n')
        self.is_prog = False
        self.last_wr = ''
        super(Logger, self).warning(msg, **kwargs)

    def error(self, msg, **kwargs):
        if self.level > 40:
            return
        if self.is_prog:
            out.write('\n')
        self.is_prog = False
        self.last_wr = ''
        super(Logger, self).error(msg, **kwargs)

    def critical(self, msg, **kwargs):
        if self.level > 50:
            return
        if self.is_prog:
            out.write('\n')
        self.is_prog = False
        self.last_wr = ''
        super(Logger, self).exception(msg, **kwargs)


if __name__ == '__main__':
    log = Logger(levl=logging.DEBUG)

    i_max = 1000000

    for i in range(i_max):
        log.progress('Counter', i, i_max)
