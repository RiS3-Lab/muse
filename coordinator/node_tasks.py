#/usr/bin/env python2
import os
import sys
import logging
import moriarty
import subprocess
from celery import Celery

logging.basicConfig()
l = logging.getLogger("node_tasks")
l.setLevel("INFO")


def get_redis():
    if not _env_is_sane():
        sys.exit(-1)
    with open(REDIS_FILE, 'r') as f:
        r = f.readline().rstrip('\n')
    l.debug("redis url: %s", r)
    return r


def _env_is_sane():
    if not os.path.exists(REDIS_FILE):
        l.error("Redis config file: %s does not exists", REDIS_FILE)
        return False
    return True

def setup_local_env(obj_dir, target_dir):
    #just copy the dir which contains everything here
    #obj_dir should locate in xlabfs
    args = ["cp", "-a", obj_dir, target_dir]
    subprocess.call(args)
    local_dir = os.path.join(target_dir, os.path.basename(os.path.normpath(obj_dir)))
    l.info("local directory: %s", local_dir)
    return local_dir

def run_moriarty(new_env, cfg):
    args = ["python", "./moriarty.py", "-t", new_env, "-c", cfg]
    subprocess.call(args)

REDIS_FILE=".redis_config"
BROKER_URL = get_redis()
app = Celery('node_tasks', broker=BROKER_URL)
app.conf.broker_transport_options = {'master_name':"mymaster"}
app.conf.task_routes = {'node_tasks.launch_campaign':{'queue': 'moriarty'}}

@app.task
def launch_campaign(obj_dir, cfg):
    l.info("setting up local env")
    l.info("source directory: %s config:%s", obj_dir, cfg)
    TMP_TARGET_DIR="/home"
    new_env = setup_local_env(obj_dir, TMP_TARGET_DIR)
    run_moriarty(new_env, cfg)
