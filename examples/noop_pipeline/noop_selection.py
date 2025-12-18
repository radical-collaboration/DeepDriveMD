#!/usr/bin/env python
import argparse
import time

def main(sleep_time=20):

    if sleep_time != 0:
        time.sleep(sleep_time)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sleep_time', type=int, default=0)
    args = parser.parse_args()
    print('nasha', int(args.sleep_time))
    main(int(args.sleep_time))
