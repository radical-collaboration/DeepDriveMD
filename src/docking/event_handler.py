import os
import argparse
import time

base_path="/gpfs/alpine/proj-shared/lrn005/RLDock_watch/"

def arg_conf():
    parser = argparse.ArgumentParser(description="Event handler for RLDock container")
    parser.add_argument("-i", "--input", help="input pdb file")
    parser.add_argument("-o", "--output", help="output score text file")
    args = parser.parse_args()
    return args

def new_event(args):
    temp_fname = "first-run"
    cmd_line = "runner.py -i {} -o {}".format(args.input, args.output)
    with open(base_path + temp_fname, "w") as f:
        f.write(cmd_line)
    return

def wait_async_result(target):
    while True:
        if not os.path.exists(target):
            time.sleep(10)
        else:
            break


if __name__ == "__main__":
    args = arg_conf()
    new_event(args)
    wait_async_result(args.output)

