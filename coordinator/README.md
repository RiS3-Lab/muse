# ConcFuzzer 
The coordinator module of ConcFuzzer, ConcFuzzer projects goes into this repository

## Getting Started

This README will use JPEG as an example showing how to prepare the targets and run  ConcFuzzer on them

### Prerequisties
The ROOT is usually baidu/ApolloSecurity/fuzz/
```
export ROOT=baidu/ApolloSecurity/fuzz/
```
Install psutil
```
sudo pip in stall psutil
```
Build and install the toolchain of ConcFuzzer
```
bash [ROOT]/BuildToolchain/build_toolchain
```
### Prepare the target (JPEG in this example)

Download jpeg tar file from
```
wget http://www.ijg.org/files/jpegsrc.v9c.tar.gz
```
Extract the target files
```
tar -xvf jpegsrc.v9c.tar.gz
```
Compile the target with wllvm
```
cd jpeg
export LLVM_COMPILER=clang
CC=wllvm ./configure
make -j8
```
Extract the bytecode file 
```
extract-bc ./.lib/djpeg 		#./.lib/djpeg.bc is generated
```
Instrumenting the bytecode file with AFL to get the binary
```
afl-clang-fast -Wl,-ljpeg -o djpeg_afl ./.lib/djpeg.bc
```
If your libjpeg.so is not compatable with the target version, link using the library in this folder
```
afl-clang-fast -Xlinker ./.lib/libjpeg.so -o jpeg_afl ./.lib/djpeg.bc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[path_to_jpeg]/.lib/
```
### Configure ConcFuzzer
Create a target folder (e.g., jpeg) and copy the two target files into the folder 
```
mv ./.lib/djpeg.bc [ROOT]/coordinator/jpeg/
mv ./djpeg_afl [ROOT]/coordinator/jpeg/
```
Create your own copy of fuzz.cfg from fuzz.cfg.template and also copy into the target folder
Modify these configure parameters
```
target_bin=@target/djpeg_afl -outfile /dev/null
target_bc=@target/djpeg.bc -outfile /dev/null
root=/apollo/src/baidu/ApolloSecurity/fuzz/AFL/
heuristics=san-guided   #also can be using random or rareness
```
## Run ConcFuzzer
Providing essential arguments to the coordinator scripts
```
python moriarty.py -t jpeg -c jpeg/fuzz.cfg 
```







