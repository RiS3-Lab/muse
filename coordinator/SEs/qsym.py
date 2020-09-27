import ConfigParser
import multiprocessing
import subprocess
import os
import sys
import utils
import shutil
import signal
import tempfile
from utils import bcolors
from utils import mkdir, mkdir_force
import qsym_minimizer as minimizer
from qsym_executor import Executor

DEFAULT_TIMEOUT = 90
TARGET_FILE = utils.AT_FILE

def se_info(s):
    print bcolors.HEADER+"[QSYM-Info]"+bcolors.ENDC," {0}".format(s)

def get_afl_cmd(fuzzer_stats):
    with open(fuzzer_stats) as f:
        for l in f:
            if l.startswith("command_line"):
                # format= "command_line: [cmd]"
                return l.split(":")[1].strip().split()

class Qsym:
    def __repr__(self):
        return "SE Engine: QSYM Concolic Explorer"

    def __init__(self, config, target):
        self.jobs = {}
        self.started_jobs = set()
        self.config = config
        self.target = target
        self.get_config()
        self.pid_ctr = 0
        self.minimizer = None
        self.make_dirs()

    def init_minimizer(self):
        if self.minimizer is not None:
            return
        cmd, afl_path, qemu_mode = self.parse_afl_stats()
        self.minimizer = minimizer.TestcaseMinimizer(
            cmd, afl_path, self.seed_dir, qemu_mode)

    def parse_afl_stats(self):
        cmd = get_afl_cmd(os.path.join(self.afl_dir, "fuzzer_stats"))
        assert cmd is not None
        index = cmd.index("--")
        return cmd[index+1:], os.path.dirname(cmd[0]), '-Q' in cmd

    @property
    def bitmap(self):
        return os.path.join(self.seed_dir, "bitmap")

    @property
    def afl_dir(self):
        return os.path.join(self.sync_dir, "slave_000001")

    def my_in_dir(self, counter):
        return os.path.join(self.seed_dir, counter)

    def my_sync_dir(self, instance):
        return os.path.join(self.sync_dir, instance, "queue")


    def get_config(self):
        config = ConfigParser.ConfigParser()
        config.read(self.config)
        self.name = config.get("qsym conc_explorer", "name")
        self.cmd = config.get("qsym conc_explorer", "cmd").replace("@target", self.target).split(" ")
        # store the selected inputs from fuzzer queue to be explored by qsym
        self.seed_dir = config.get("qsym conc_explorer", "qsym_seed_dir").replace("@target", self.target)
        self.sync_dir = config.get("moriarty", "sync_dir").replace("@target", self.target)

        try:
            self.max_time_per_seed = config.get("qsym conc_explorer", "max_time_per_seed")
        except Exception:
            self.max_time_per_seed = DEFAULT_TIMEOUT
        try:
            self.max_mem = config.get("klee conc_explorer", "max_memory")
        except Exception:
            self.max_mem = str(1024*1024*20) # in kbytes

    def make_dirs(self):
        mkdir_force(self.seed_dir)

    # cov_file is dummy parameter
    def run(self, input_id_map_list, cov_file):
        """
            -create seed-out-dir
            For each input,
                -convert ktest move to seed-out-dir
            -create sync dir
            -build cmd
            -create new process job
        """
        pid = self.get_new_pid()
        qsym_seed_dir = self.my_in_dir(str(pid))
        mkdir_force(qsym_seed_dir)

        se_info("{0} activated. input list : {1}".format(self, [x['input'] for x in  input_id_map_list]))
        se_info("{0} activated. input score : {1}".format(self, [x['score'] for x in  input_id_map_list]))
        se_info("{0} activated. input size: {1}".format(self, [x['size'] for x in  input_id_map_list]))

        # sync previously generated seeds
        self.sync_gen_seeds()

        # launch qsym for each inputs in my_in_dir
        for input_id_map in input_id_map_list:
            #QSYM does not support batch mode
            assert len(input_id_map_list) <= 1

            # print input_id_map
            afl_input = input_id_map['input']
            qsym_seed = os.path.join(qsym_seed_dir, afl_input.split("/")[-1])
            shutil.copy2(afl_input, qsym_seed)
            if not os.path.exists(qsym_seed):
                se_info("no seed created: " + qsym_seed)
                continue

            #--create sync_dir for new qsym instance
            key = "qsym_instance_conc_" + str(pid).zfill(6)
            new_sync_dir = self.my_sync_dir(key)
            mkdir_force(new_sync_dir)

            # temp dir to store tmp genearated seeds
            # filtered seeds will be transfer to new_sync_dir
            tmp_dir = tempfile.mkdtemp()
            mkdir_force(tmp_dir)

            #--build qsym instance cmd
            q, qsym_cmd = self.build_cmd(qsym_seed, tmp_dir, self.bitmap)
            print ' '.join(qsym_cmd)
            # q.run(self.max_time_per_seed)

            #--construct process meta data, add to jobs list
            kw = {'stdin':q.stdin, 'mem_cap': self.max_mem, 'use_shell':True}
            p = multiprocessing.Process(target=utils.qsym_exec_async, args=[qsym_cmd], kwargs=kw)
            p.daemon = True
            task_st = {}
            task_st['instance'] = p
            task_st['sync_dir'] = new_sync_dir
            task_st['cmd'] = qsym_cmd
            task_st['tmp_dir'] = tmp_dir
            task_st['qsym'] = q
            task_st['seed_index'] = 0
            task_st['synced'] = False
            task_st['key'] = key
            task_st['processed'] = False
            self.jobs[pid] = task_st

        for pid, task in self.jobs.iteritems():
            try:
                if pid not in self.started_jobs:
                    task['instance'].start()
                    task['real_pid'] = task['instance'].pid
                    self.started_jobs.add(pid)

            except Exception:
                pass
        return (key, [x['input'] for x in input_id_map_list])

    def build_cmd(self, cur_input, cur_output, bitmap):
        q = Executor(self.cmd, cur_input, cur_output, bitmap=bitmap, argv=["-l", "1"])
        cmd = q.gen_cmd(self.max_time_per_seed)
        return q, cmd

    def sync_gen_seeds(self):
        self.init_minimizer()

        # copy the generated inputs back
        for pid, task in self.jobs.iteritems():
            if task['synced']:
                continue
            print "syncing ", task['sync_dir']
            print "syncing ", task['key']
            task['synced'] = True
            qsym = task['qsym']
            target_dir = task['sync_dir']
            index = task['seed_index']

            # for testcase in qsym.get_testcases():
            #     filename = os.path.join(target_dir, "id:%06d:src:%s" % (index, 'qsym'))
            #     index += 1
            #     se_info("moving %s to %s" % (testcase, filename))
            #     shutil.move(testcase, filename)

            num_testcase = 0
            for testcase in qsym.get_testcases(task['sync_dir']):
                # print testcase
                if not self.minimizer.check_testcase(testcase):
                    # Remove if it's not interesting testcases
                    # os.unlink(testcase)
                    continue
                target = os.path.basename(testcase)
                filename = os.path.join(target_dir, "id:%06d:src:%s" % (index, task['key']+','+target))
                index += 1
                # se_info("moving %s to %s" % (testcase, filename))
                shutil.move(testcase, filename)
                se_info("Creating: %s" % filename)

            # remove the tmp dir
            shutil.rmtree(task['tmp_dir'])

    def alive(self):
        alive = False
        #This call is to activate something (sorry i don't remember now :-/)
        multiprocessing.active_children()
        for pid in [self.jobs[x]['real_pid'] for x in self.jobs]:
            try:
                os.kill(pid, 0)
                print "conc_explorer pid: {0} is alive".format(pid)
                alive = True
            except Exception:
                pass
                # print "conc_explorer pid: {0} not alive".format(pid)

        return alive

    def stop(self):
        """
        Terminate all jobs,
        you could have more fine-grained control by extending this function
        """
        se_info("{0} deactivated".format(self))
        self.sync_gen_seeds()
        for pid, task in self.jobs.iteritems():
            if task['processed']:
                continue
            task['processed'] = True
            se_info("Terminting qsym instance: {0} {1} real pid:{2}".format(pid, task['instance'], task['real_pid']))
            utils.terminate_proc_tree(task['real_pid'])
            self.jobs[pid]['processed'] = True

    def get_new_pid(self):
        self.pid_ctr += 1
        return self.pid_ctr

    def terminate_callback(self):
        """called when SIGINT and SIGTERM"""
	pass

    def periodic_callback(self):
        """called every 1 hour"""
	pass

