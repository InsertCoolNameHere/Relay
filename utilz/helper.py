from time import time
import logging

def fancy_logging(msg, hostname="", logger = None):
    strm = ""
    strm += "%s:====================================\n" % hostname
    strm += "%s: %s : TIME: %s"%(hostname, msg, str(time()))
    print(strm)
    if logger:
        logger.info(strm)


def fancy_wrapup(msg, hostname="", logger = None):
    strm = ""
    strm += "%s:====================================\n" % hostname
    strm += "%s: %s : TIME: %s" % (hostname, msg, str(time()))
    strm += "%s:*************************************\n" % hostname
    print(strm)
    if logger:
        logging.info(strm)
