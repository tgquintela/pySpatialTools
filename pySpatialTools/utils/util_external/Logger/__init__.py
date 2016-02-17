
"""
Log functionalities in order to save every comment in a log.

TODO:
-----
- Continue the writting in the point before not in the next line.
"""

from datetime import datetime
from os.path import exists

mark_dt_str = '\n'+'='*80+'\n'+'%s'+'\n'+'='*80+'\n'+'='*80+'\n\n'


class Logger:
    """
    Logger class provides a functionality to write in a logfile and display in
    screen a given message.
    """

    def __init__(self, logfile):
        self.logfile = logfile
        if not exists(logfile):
            initial = self.mark_datetime('Creation of the logfile')
            self.write_log(initial, False)

    def write_log(self, message, display=True):
        """Function to write and/or display in a screen a message."""
        if display:
            print message
        append_line_file(self.logfile, message)  # +'\n')

    def mark_datetime(self, message=''):
        """Function to write the datetime in this momment."""
        dtime = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        message = message+': ' if len(message) > 0 else message
        m = ' ' + message + dtime + ' '
        n = ((80-len(m))/2)
        m = n*'='+m+n*'='
        m = m+'=' if len(m) != 80 else m
        m = mark_dt_str % m
        return m

    def get_datetime(self):
        """Easy and quick way to get datetime."""
        dtime = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        return dtime


def append_line_file(filename, line):
    f = open(filename, 'a')
    f.write(line)
    f.close()
