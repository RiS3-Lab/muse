#!/usr/bin/fish

pushd AFL
make

pushd llvm_mode
make
popd

pushd libdislocator
make
popd
popd

pushd qsym
echo 0|sudo tee /proc/sys/kernel/yama/ptrace_scope
./setup.sh

virtualenv venv
source venv/bin/activate.fish
pip install .

deactivate
popd



