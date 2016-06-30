import datetime
import locale

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
        ms = diff.seconds * 1000 + diff.microseconds / 1000

        if msg is not None:
            locale.setlocale(locale.LC_ALL, 'en_US')
            formatted_ms = locale.format('%d', ms, grouping=True)
            print('%s: %s ms (%i)' % (msg, formatted_ms, length))
        return ms
