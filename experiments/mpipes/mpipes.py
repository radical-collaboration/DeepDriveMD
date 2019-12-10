from radical import entk
import os

hostname = os.environ.get('RMQ_HOSTNAME', 'localhost')
port = int(os.environ.get('RMQ_PORT', 5672))
os.environ["RADICAL_LOG_TGT"] = "radical.entk.mpipe.log"
os.environ["RADICAL_LOG_LVL"] = "DEBUG"

def gen_task(name, executable, params=None):

    t = entk.Task()
    t.name = name
    t.executable = executable
    t.arguments = params if isinstance(params, list) else [params]
    t.cpu_reqs = { 'processes': 1,
            'process_type': None,
            'threads_per_process': 4,
            'thread_type': 'OpenMP'}
    t.gpu_reqs = { 'processes': 1,
            'process_type': None,
            'threads_per_process': 1,
            'thread_type': 'CUDA'
            }
    return t

def gen_mpipe(loop_count=10, conc_count=3, exec_second=10):
    p = entk.Pipeline()
    for i in range(loop_count):
        s = entk.Stage()
        for j in range(conc_count):
            t = gen_task('multi-tasks-in-multi-pipes-{:03}-{:03}'.format(i, j),
                    '/bin/sleep', exec_second)
            s.add_tasks(t)
        p.add_stages(s)
    return p

def main():
    appman = entk.AppManager(hostname=hostname, port=port)

    res_dict = {
        'resource': 'ornl.summit',
        'queue': 'batch',
        'schema': 'local',
        'walltime': 10,
        'cpus': 6,
        'gpus': 6,
        'project': 'CSC393'
    }

    appman.resource_desc = res_dict
    p1 = gen_mpipe(10, 2, 10)
    p2 = gen_mpipe(10, 2, 100)
    p3 = gen_mpipe(10, 2, 1000)
    appman.workflow = [p1, p2, p3]
    appman.run()

if __name__ == '__main__':
    main()

    
