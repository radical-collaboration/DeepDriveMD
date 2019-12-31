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
            'cpus'         : 168*1.5
            }

    appman = prepare_entk(res_dict)
    ps = []
    p1 = entk.Pipeline()
    s1 = set_ratio({"40:10:10": "1000,100,10"})
    p1.add_stages(s1)
    for i in range(3):
        pn = entk.Pipeline()
        sn = set_ratio({"10:10": "100,10"})
        pn.add_stages(sn)
        ps.append(pn)

    appman.workflow = [p1] + ps
    appman.run()

