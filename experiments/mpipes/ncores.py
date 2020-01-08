from radical import pilot as rp, utils as ru
from radical import entk
import os

def prepare_entk(rd):

    hostname = os.environ.get('RMQ_HOSTNAME', 'two.radical-project.org')
    port = int(os.environ.get('RMQ_PORT', 33239))

    appman = entk.AppManager(hostname=hostname, port=port)
    appman.resource_desc = rd
    return appman


def gen_tasks(loop_count=1, exec_second=10):
   
    ts = []
    for i in range(int(loop_count)):
        t = entk.Task()
        t.pre_exec = ["date"]
        t.pre_exec += ["echo 'for i in $(seq {}); do /usr/bin/sleep 1; date; done' > a.sh".format(exec_second)]
        t.executable = 'bash a.sh' 
        t.post_exec = ["date"]
        t.cpu_reqs = {'processes': 1,
                'process_type': None,
                'thread_type' : "OpenMP",
                'threads_per_process': 4 }
        ts.append(t)
    return ts

def set_ratio(prop_per_task={"1:1:1":"10,100,1000"}):
    ntask = []
    for k, v in prop_per_task.items():
        for loop_count, exec_second in zip(k.split(":"), v.split(",")):
            ntask += gen_tasks(loop_count, exec_second)
    s = entk.Stage() 
    for task in ntask:
        s.add_tasks(task)
    return s


if __name__ == '__main__':

    res_dict = {
            'resource'      : "ornl.summit",
            'walltime'       : 30,
            'project'       : "LRN005",
            'queue'         : "batch",
            'cpus'         : 168*3
            }


    res_set = {"100:100:52":"1000,100,10"}
    res_set = {"100:100:10":"1000,100,10"}
    res_set = {"100:66:2":"1000,100,10"}
    res_set = {"100:25:1":"1000,100,10"}
    tcpus = 0
    ntasks = 100
    pipes = []
    for k, v in res_set.items():
        for cpus, exec_second in zip(k.split(":"), v.split(",")):
            cpus = int(cpus)
            p = entk.Pipeline()
            for i in range(0, ntasks, cpus):
                if ntasks - i < cpus:
                    ptask = ntasks - i
                else:
                    ptask = cpus
                s = set_ratio({str(ptask):str(exec_second)})
                #print(ptask,exec_second)
                p.add_stages(s)
            tcpus += cpus
            pipes.append(p)
    #print(tcpus)
    #import sys
    #sys.exit()
    res_dict['cpus'] = tcpus * 4
    appman = prepare_entk(res_dict)
    appman.workflow = pipes
    appman.run()

