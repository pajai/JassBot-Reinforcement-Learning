import datetime

class Chrono:

    def __init__(self):
        self.start_times = []

    def start(self):
        self.start_times.append(datetime.datetime.now())

    def stop(self, msg = None):
        length = len(self.start_times)
        if length == 0:
            raise Exception('stop should not be called before start')

        start = self.start_times.pop()
        diff = (datetime.datetime.now() - start)
        ms = diff.microseconds / 1000

        if msg is not None:
            print('%s: %i ms (%i)' % (msg, ms, length))
        return ms
