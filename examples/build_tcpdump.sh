#! /bin/bash

# get source code
git clone https://github.com/the-tcpdump-group/tcpdump.git
git clone https://github.com/the-tcpdump-group/libpcap.git

pushd libpcap
# generate whole program bc
CC=wllvm LLVM_COMPILER=clang CFLAGS="-fsanitize=undefined -g" ./configure  --enable-shared=no
sed -i -e '/-fpic/ s/pic/PIC/' Makefile
LLVM_COMPILER=clang make -j$(nproc)

popd
pushd tcpdump
CC=wllvm LLVM_COMPILER=clang CFLAGS="-fsanitize=undefined -g" LDFLAGS="-lubsan" ./configure
LLVM_COMPILER=clang make -j$(nproc)

# extract bc
extract-bc tcpdump

#set up fuzzing work dir
mkdir obj-muse && pushd obj-muse && cp ../tcpdump.bc .

# get binary for qsym
cp ../tcpdump .

#generate binary to be fuzzed and target bc to be analyzed
apt-get install -y libdbus-1-dev
~/work/muse/AFL/afl-clang-fast tcpdump.bc -o afl-tcpdump -lcrypto -libverbs  -ldbus-1 -fsanitize=undefined

#generate instrumented binary to replay by coordinator 
~/work/muse/DynInstr/dyn-instr-clang tcpdump.bc -o dyninst-tcpdump -lcrypto -libverbs  -ldbus-1 -fsanitize=undefined

#run svf analyzer (llvm-4.0) on the target bc
# running this command gives the following output files
# -PROG.edge file records each basic block and its outgoing edge IDs
# -PROG.reach.cov file records each BB ID and how many basic blocks it can reach
# -PROG.reach.bug file records each BB ID and how many Sanitizer Lbales it can reach
python ~/work/muse/svf/SVF/dma_wrapper.py -fspta dyninst-tcpdump.bc -o tcpdump.reach -edge tcpdump.edge

popd
echo "Preparation done, please edit the config file and prepare the seeding inputs for fuzzing"
cp ~/work/muse/coordinator/configs/fuzz.tcpdump.cfg fuzz.cfg
mkdir -p tcpdump/obj-muse/in &&  cp tcpdump/tests/02-sunrise-sunset-esp.pcap tcpdump/obj-muse/in
echo "target direction: tcpdump/obj-muse"
echo "config template: tcpdump/obj-muse/fuzz.cfg"
