import os
from datetime import datetime


def save_config(save_root, cfg):
    now = cfg.SYSTEM.EXP_NAME
    dir_name = 'runs'
    dir_name = os.path.join(save_root, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fn = cfg.SYSTEM.EXP_NAME
    log_dir = os.path.join(dir_name, fn)

    dir_name = cfg.SYSTEM.EXP_NAME
    dir_name = os.path.join(save_root, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    chk_dir = os.path.join(dir_name, 'chkpt')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(chk_dir):
        os.makedirs(chk_dir)

    with open(os.path.join(dir_name, 'config.txt'), 'w') as f:
        f.write(str(cfg))

    return log_dir, chk_dir
