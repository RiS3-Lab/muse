################################################################
#  MEUZZ:Smart Seed Scheduling For Hybrid Fuzzing - Dockerfile #
#                 (In RAID, 2020)                              #
#                                                              #
#        Author: Yaohui Chen <yaohway@gmail.com>               #
#        Computer Science@Northeastern University              #
#                                                              #
#  This file can be distributed under the MIT License.         #
#  See the LICENSE.TXT for details.                            #
################################################################

From ubuntu:16.04
LABEL maintainers="Yaohui Chen"
LABEL version="0.1" 


RUN           apt-get update && apt-get install -y \
              wget            \
              vim             \
              git             \
              python          \  
              build-essential \
              autoconf        \
              python-pip      \
              python-dev      \
              python-tk       \
              cmake           \
              sudo



#RUN           pip install --upgrade pip
RUN           wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
RUN           python get-pip.py
RUN           pip install psutil
RUN           pip install wllvm
RUN           pip install virtualenv
RUN           pip install numpy 
RUN           pip install pandas 
RUN           pip install scikit-learn 
RUN           pip install matplotlib 
RUN           pip install seaborn 
RUN           pip install termcolor 

RUN           mkdir  -p      /root/work/muse
COPY          DynInstr       /root/work/muse/DynInstr
COPY          patches        /root/work/muse/patches
COPY          svf            /root/work/muse/svf
COPY          AFL            /root/work/muse/AFL
COPY          qsym           /root/work/muse/qsym
COPY          coordinator    /root/work/muse/coordinator        
COPY          examples       /root/work/muse/examples      
COPY          Docker/docker_build_muse.sh /root/work/muse
COPY          Docker/build_qsym.sh /root/work/muse/build_qsym.sh

WORKDIR       /root/work
RUN           cd /root/work/muse && ./docker_build_muse.sh
RUN           cd /root/work/muse && ./build_qsym.sh
  
