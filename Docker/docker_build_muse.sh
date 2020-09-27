#!/bin/bash
PROJ=muse
SOFTWARE_DIR=~/softwares
WORK_DIR=~/work/

function dir_check {
    if [ -d $2 ]
    then
        echo "[DIRECTORY CHECK]" $1 " pass"
    else
        echo "[DIRECTORY CHECK] Error: missing "$1
        exit -1
    fi
}
################Some sanity check, everything we need is in the WORK_DIR/muse directory
    dir_check $PROJ       $WORK_DIR/$PROJ
    dir_check svf         $WORK_DIR/$PROJ/svf
    dir_check AFL         $WORK_DIR/$PROJ/AFL
    dir_check qsym        $WORK_DIR/$PROJ/qsym
    dir_check DynInstr    $WORK_DIR/$PROJ/DynInstr
    dir_check coordinator $WORK_DIR/$PROJ/coordinator
    dir_check patches     $WORK_DIR/$PROJ/patches
    dir_check examples 	  $WORK_DIR/$PROJ/examples


################Download packages, might need to manually compile and install

    rm -rf $SOFTWARE_DIR
    mkdir $SOFTWARE_DIR

    PROG=llvm-3.8
    echo "Downloading $PROG"
    cd $SOFTWARE_DIR
    wget http://releases.llvm.org/3.8.0/clang-tools-extra-3.8.0.src.tar.xz
    wget http://releases.llvm.org/3.8.0/compiler-rt-3.8.0.src.tar.xz
    wget http://releases.llvm.org/3.8.0/llvm-3.8.0.src.tar.xz
    wget http://releases.llvm.org/3.8.0/cfe-3.8.0.src.tar.xz .

    echo "Download finished"
    echo "Now go to $SOFTWARE_DIR and compile then install them"
    cd $SOFTWARE_DIR

################Installation script

    sudo apt-get install -y zlib1g-dev libncurses5-dev libdbus-1-3 flex bison libdbus-1-dev libibverbs-dev
    sudo apt-get install -y build-essential libssl-dev libcurl4-gnutls-dev libexpat1-dev gettext unzip

    #install llvm-3.8 for qsym and afl
    PROG=llvm-3.8
    cd $SOFTWARE_DIR
    tar xvf cfe-3.8.0.src.tar.xz
    tar xvf llvm-3.8.0.src.tar.xz
    tar xvf compiler-rt-3.8.0.src.tar.xz
    tar xvf clang-tools-extra-3.8.0.src.tar.xz

    mv llvm-3.8.0.src llvm-3.8
    mv cfe-3.8.0.src llvm-3.8/tools/clang
    mv clang-tools-extra-3.8.0.src llvm-3.8/tools/clang/extra
    mv compiler-rt-3.8.0.src llvm-3.8/projects/compiler-rt

    patch -p0 < $WORK_DIR/muse/patches/clang-3.8.patch
    patch -p0 < $WORK_DIR/muse/patches/llvm-3.8-asan.patch


    cd $SOFTWARE_DIR/llvm-3.8
    mkdir build
    cd $SOFTWARE_DIR/llvm-3.8/build
    cmake -DLLVM_ENABLE_RTTI:BOOL=ON -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_BUILD_TYPE=Release ..
    make install -j$(nproc)

    PROG=afl
    echo "Installing $PROG"
    cd $WORK_DIR/$PROJ/AFL
    sed -i -e '/MAP_SIZE_POW2       16/ s/MAP_SIZE_POW2.*/MAP_SIZE_POW2       20/' config.h
    make
    cd $WORK_DIR/$PROJ/AFL/llvm_mode
    make

    PROG=DynInstr
    echo "Installing $PROG"
    cd $WORK_DIR/$PROJ/DynInstr
    make

    #build svf
    PROG=svf
    cd $WORK_DIR/$PROJ/svf
    git checkout savior
    
    cd SVF
    #download llvm-4.0 for SVF (only used locally)
    wget https://releases.llvm.org/4.0.0/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
    tar xf clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz

    export LLVM_DIR=$(pwd)/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04
    export PATH=$PATH:$(pwd)/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04
    cd Release-build
    cmake ..
    make -j$(nproc)
    cd .. && ln -sf $(pwd)/Release-build/bin/dma_wrapper.py dma_wrapper.py

