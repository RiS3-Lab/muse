# MEUZZ Fuzzer 

## Clone the repo

git clone --recurse-submodules git@github.com:RiS3-Lab/muse.git


## Work Flow

1. SRC + Sanitizer (use wllvm) => prog + prog.bc
2. prog.bc (use afl-clang-fast) => afl-prog
3. prog.bc (use dyn-instr-clang) => dyninst-prog + dyninst-prog.bc
4. dyninst-prog.bc (use DMA) => prog.edge + prog.reach.bug + prog.reach.cov

Eventually, update config file 
AFL fuzz with *afl-prog*
- fill in [moriarty]
    - target_bin (remember to put in proper cli options as well!)
QSYM run with *prog*
- fill in [qsym conc_explorer]
    - cmd (remember to put in proper cli options as well!)
Coordinator use *dyninst-prog* + *prog.edge* + *prog.reach.bug* + *prog.reach.cov* to replay and collect data
- fill in [auxiliary info]
    - replay_prog_cmd (remember to put in proper cli options as well!)
    - bbl_bug_map
    - bbl_cov_map
    - pair_edge_file


To see some examples on how to configure muse, please look at the [examples](./examples) directory.

## How to build Muse

### Build with Docker
```
$ curl -fsSL https://get.docker.com/ | sudo sh

$ sudo usermod -aG docker [user_id]

$ docker run ubuntu:16.04

Unable to find image 'ubuntu:16.04' locally
16.04: Pulling from library/ubuntu
Digest: sha256:e348fbbea0e0a0e73ab0370de151e7800684445c509d46195aef73e090a49bd6
Status: Downloaded newer image for ubuntu:16.04

$ docker build -t muse .

$ docker images

REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
muse                latest              687322eff8f3        29 minutes ago      30.5GB
ubuntu              16.04               f975c5035748        10 days ago         112MB

```
Once the build has been successful, lunch the Docker image 
to test out MUSE.


```
# use --privileged flag to allow docker to mount a ramdisk

$ docker run --cap-add=SYS_PTRACE --priviledged -it muse:latest /bin/bash
```

## Citation
If your research find one or several components of savior useful, please cite the following paper:
```
@inproceedings{meuzz,
  author    = {Chen, Yaohui and
              Ahmadi, Mansour and
              Mirzazade farkhani, Reza and
              Wang, Boyu and
              Lu, Long},
  title     = {{MEUZZ}: Smart Seed Scheduling for Hybrid Fuzzing},
  booktitle = {Proceedings of the 23rd International Symposium on Research in Attacks, Intrusions and
               Defenses},
  series = {RAID'20},
  year      = {2020},
  month={October}
}
```


## Q&A

0. why pass -fsanitize=address,undefined when preparing bc file with wllvm?

Answer: for 2 reasons. First, UBSAN is only available at front-end; Second, DynInstr needs the input BC to be instrumented with sanitizer labels to guide its instrumentations.

1. why use llvm-3.8 and llvm-4.0

Answer: QSYM requires 3.8 while SVF requires 4.0

2. How to specify stdin and file input type

Anser: Use the @@ to specify if input is passed from file in the cli option (similar to AFL). See example configs under coordinator/configs 

