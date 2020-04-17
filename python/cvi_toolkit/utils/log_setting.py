import logging


log_name = dict()
def setup_logger(name, log_level="INFO"):
    if name in log_name:
        return log_name[name]

    formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S', fmt='%(asctime)s->%(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    log_name[name] = logger
    return logger