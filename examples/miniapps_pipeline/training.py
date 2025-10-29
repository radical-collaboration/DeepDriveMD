#!/usr/bin/env python

import io, os, sys, socket
import time
import argparse
import wfMiniAPI.kernel as wf

def parse_args():
    parser = argparse.ArgumentParser(description='Exalearn_miniapp_training')
    parser.add_argument('--num_epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--device', default='gpu',
                        help='Wheter this is running on cpu or gpu')
    parser.add_argument('--phase', type=int, default=0,
                        help='the current phase of workflow, phase0 will not read model')
    parser.add_argument('--data_root_dir', default='./',
                        help='the root dir of gsas output data')
    parser.add_argument('--model_dir', default='./',
                        help='the directory where save and load model')
    parser.add_argument('--num_sample', type=int, default=100,
                        help='num of samples in matrix mult')
    parser.add_argument('--num_mult', type=int, default=10,
                        help='number of matrix mult to perform')
    parser.add_argument('--dense_dim_in', type=int, default=12544,
                        help='dim for most heavy dense layer, input')
    parser.add_argument('--dense_dim_out', type=int, default=128,
                        help='dim for most heavy dense layer, output')
    parser.add_argument('--mat_size', type=int, default=5000,
                        help='the matrix with have size of mat_size * mat_size, should be the same as it is in simulation')
    parser.add_argument('--preprocess_time', type=float, default=20.0,
                        help='time for doing preprocess')
    parser.add_argument('--read_size', type=int, default=0,
                        help='size of bytes read from disk')
    parser.add_argument('--write_size', type=int, default=3500000,
                        help='size of bytes written to disk, -1 means write data to disk once')
    parser.add_argument('--instance_index', type=int, required=True,
                        help='use to distinguish different training task. Should be from 0~n-1')

    args = parser.parse_args()

    return args


def main():

    print("Temp for Darshan, ml, PID = {}, hostname = {}".format(os.getpid(), socket.gethostname()))
    start_time = time.time()

    args = parse_args()
    print(args)

    root_path = args.data_root_dir + '/phase{}'.format(args.phase) + '/'
    print("root_path for data = ", root_path)

    device = args.device

#FIXME: Commented out since it can not load cupy... Weird!    
#    if device == 'gpu':
#        print("gpu id is {}".format(cupy.cuda.runtime.getDeviceProperties(0)['uuid']))

    try:
        import cupy
        device = "gpu"
    except ImportError:
        device = "cpu"

    wf.readNonMPI(args.read_size, root_path, args.instance_index)
    wf.sleep(args.preprocess_time)
    wf.generateRandomNumber(device, args.num_sample * args.dense_dim_in)
    wf.generateRandomNumber(device, args.dense_dim_in * args.dense_dim_out)
    if device == 'gpu':
        wf.dataCopyH2D(args.dense_dim_in * args.dense_dim_out)


    for epoch in range(args.num_epochs):
        tt = time.time()
        if device == "gpu":
            wf.dataCopyH2D(args.num_sample * args.dense_dim_in)
        print("epoch is {}, data movementi (CPU->GPU) takes {}".format(epoch, time.time() - tt))
        tt = time.time()
        for ii in range(args.num_mult):
            wf.matMulGeneral(device, [args.num_sample, args.dense_dim_in], [args.dense_dim_in, args.dense_dim_out], ([1], [0]))
            wf.axpy(device, args.dense_dim_in * args.dense_dim_out)
        print("epoch is {}, mult takes {}".format(epoch, time.time() - tt))
        tt = time.time()

    if device == 'gpu':
        wf.dataCopyD2H(args.dense_dim_in * args.dense_dim_out)
    wf.writeNonMPI(args.write_size, root_path, args.instance_index)

    end_time = time.time()
    print("Total training time is {} seconds".format(end_time - start_time))

if __name__ == '__main__':
    main()
