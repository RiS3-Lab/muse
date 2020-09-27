#!/bin/bash
PROJ=muse
SOFTWARE_DIR=~/softwares
WORK_DIR=~/work/

PROG=qsym
cd $WORK_DIR/$PROJ/qsym
echo 0|sudo tee /proc/sys/kernel/yama/ptrace_scope
apt-get install -y git build-essential sudo
./setup.sh
virtualenv venv
source venv/bin/activate
pip install .
deactivate
# install dependencies
sudo pip install hanging_threads
