#! /bin/bash

# get source code
wget https://www.ijg.org/files/jpegsrc.v9c.tar.gz
tar zxf jpegsrc.v9c.tar.gz

pushd jpeg-9c

# generate whole program bc
CC=wllvm LLVM_COMPILER=clang CFLAGS="-fsanitize=undefined -g" ./configure  --enable-shared=no --enable-static=yes
LLVM_COMPILER=clang make -j$(nproc)

# extract bc
extract-bc djpeg

#set up fuzzing work dir
mkdir obj-muse && pushd obj-muse && cp ../djpeg.bc .

# get binary for qsym
cp ../djpeg . 

#generate binary to be fuzzed 
~/work/muse/AFL/afl-clang-fast djpeg.bc -o afl-djpeg -fsanitize=undefined -lm

#generate instrumented binary to replay by coordinator 
~/work/muse/DynInstr/dyn-instr-clang djpeg.bc -o dyninst-djpeg -fsanitize=undefined -lm

#run svf analyzer (llvm-4.0) on the target bc
# running this command gives the following output files
# -PROG.edge file records each basic block and its outgoing edge IDs
# -PROG.reach.cov file records each BB ID and how many basic blocks it can reach
# -PROG.reach.bug file records each BB ID and how many Sanitizer Lbales it can reach
python ~/work/muse/svf/SVF/dma_wrapper.py -fspta dyninst-djpeg.bc -o djpeg.reach -edge djpeg.edge


echo "Preparation done, please edit the config file and prepare the seeding inputs for fuzzing"
cp ~/work/muse/coordinator/configs/fuzz.djpeg.cfg fuzz.cfg
cp -a ~/work/muse/AFL/testcases/images/jpeg/ in
echo "target direction: jpeg-9c/obj-muse"
echo "config template: jpeg-9c/obj-muse/fuzz.cfg"
