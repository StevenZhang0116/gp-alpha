"""
Created By Dawei Chen
Feb 2018
日志

for python 3, win/linux

使用方法：

from py3lib import dlog
dlog.init(folder)               # 设定日志所在文件夹，日志文件名为 <folder>/YYYYMMDD.log。如果不设定，则只输出到屏幕
dlog.debug(msg)                 # 输出调试信息
dlog.info(msg)
dlog.warning(msg)
dlog.error(msg, True)           # 第二个参数为True时会输出当前堆栈，默认为False
dlog.setLevel(logging.DEBUG)    # 如果不设定，默认为logging.INFO

Changelog:
20180510 dawei init增加参数，可不输出到屏幕
20180218 dawei first version
"""

import datetime, os.path, logging, getpass, platform, sys

class dlog:
    pass

class LogHandler:
    """日志类"""
    def __init__(self):
        self.logfoldername = ''
        self.logdate = 0
        self.__inited = False
        self.__errorLevel = logging.DEBUG
        self.__handler = None
        self.printscreen = True

    def __check_log_file(self):
        """检查日志文件"""
        if self.logfoldername == '':
            if self.__inited:
                pass
            else:
                if self.printscreen:
                    # host = os.getenv('COMPUTERNAME', '') if platform.system() == 'Windows' else os.getenv('HOSTNAME', '')
                    user = getpass.getuser()
                    format = logging.Formatter(u'[%s %s %s] %s' % ('%(levelname)-5.5s', '%(asctime)s', user, "%(message)s"))
                    # format = logging.Formatter(u'[%s %s %s] %s' % ('%(levelname)s', '%(asctime)s', user, "%(message)s"))
                    # format = u'[%s %s %s@%s] %s' % ('%(levelname)-5.5s', '%(asctime)s', user, host, "%(message)s")
                    self.__handler = logging.getLogger('consolelog')
                    self.__handler.setLevel(logging.INFO)
                    stream_handler = logging.StreamHandler(stream=sys.stdout)
                    stream_handler.setFormatter(format)
                    self.__handler.addHandler(stream_handler)
            self.__inited = True
        else:
            d = datetime.date.today()
            if self.logdate != d or not self.__inited or self.__handler is None:
                self.logdate = d
                # host = os.getenv('COMPUTERNAME', '') if platform.system() == 'Windows' else os.getenv('HOSTNAME', '')
                user = getpass.getuser()
                format = logging.Formatter(u'[%s %s %s] %s' % ('%(levelname)-5.5s', '%(asctime)s', user, "%(message)s"))
                # format = u'[%s %s %s@%s] %s' % ('%(levelname)-5.5s', '%(asctime)s', user, host, "%(message)s")
                datestr = self.logdate.strftime('%Y%m%d')
                self.__handler = logging.getLogger(datestr)
                self.__handler.setLevel(logging.INFO)
                logfile_handler = logging.FileHandler(os.path.join(self.logfoldername, datestr + u'.log'), encoding='utf8')
                logfile_handler.setFormatter(format)
                self.__handler.addHandler(logfile_handler)
                if self.printscreen:
                    stream_handler = logging.StreamHandler()
                    stream_handler.setFormatter(format)
                    self.__handler.addHandler(stream_handler)
                self.__inited = True
            else:
                pass

    def debug(self, msg):
        """输出调试信息
        @param msg: 要输出的信息
        """
        try:
            self.__check_log_file()
            if self.__handler is not None:
                self.__handler.debug(msg)
        except:
            pass

    def info(self, msg):
        """输出日志
        @param msg: 要输出的信息
        """
        try:
            self.__check_log_file()
            if self.__handler is not None:
                self.__handler.info(msg)
        except:
            pass

    def warning(self, msg):
        """输出警告
        @param msg: 要输出的信息
        """
        try:
            self.__check_log_file()
            if self.__handler is not None:
                self.__handler.warning(msg)
        except:
            pass

    def error(self, msg, output_stack=False):
        """输出错误
        @param msg: 要输出的信息
        @param output_stack: 如果为True, 会同时输出当前堆栈。默认为False。
        @remark 根据logging的文档，最好在except语句内才让output_stack为True，否则可能会有奇怪的错误. 见 https://stackoverflow.com/questions/5191830/how-do-i-log-a-python-error-with-debug-information
        """
        try:
            self.__check_log_file()
            if self.__handler is not None:
                if output_stack:
                    self.__handler.exception(msg)
                else:
                    self.__handler.error(msg)
        except:
            pass

    def setLevel(self, level):
        """设定输出的日志等级。低于此等级的日志将不输出。不调用本函数时，默认的等级为 logging.INFO
        @param level: 等级。从低到高为 logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR
        """
        global __errorLevel
        self.__errorLevel = level
        if self.__handler is not None:
            self.__handler.setLevel(level)

    def init_log(self, foldername, print_screen = True, attr=0o644):
        """设定日志所在文件夹，为空则不修改文件夹。日志文件名为 <folder>/YYYYMMDD.log. print_screen为是否输出到屏幕, def_attr是日志文件的属性(仅支持Linux)"""
        self.printscreen = print_screen
        if foldername != '':
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            self.logfoldername = foldername
            self.__check_log_file()

    def close(self):           
        handlers = self.__handler.handlers[:]
        for handler in handlers:
            handler.close()
            self.__handler.removeHandler(handler)

dlog = LogHandler() # default logger
debug = dlog.debug
info = dlog.info
warning = dlog.warning
error = dlog.error
init_log = dlog.init_log
setLevel = dlog.setLevel


if __name__ == '__main__':
    dlog.init_log(r'logtest')
    dlog.info('test')
    dlog.info(u'中文')
