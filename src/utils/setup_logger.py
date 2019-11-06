import logging
import os


def setup_logger(name='', save_dir='./', fn='log.txt'):
    if name is '':
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(name)
    fn = os.path.join(save_dir, fn)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(fn)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)

    logger.addHandler(fh)
    logger.addHandler(sh)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    return logger
