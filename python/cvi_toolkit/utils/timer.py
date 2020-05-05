from time import time
class Timer(object):
    def __init__(self):
        self._tstart_stack = list()

    def tic(self):
        self._tstart_stack.append(time())

    def toc(self, event_name):
        print("{}: {}s".format(event_name, time() - self._tstart_stack.pop()))