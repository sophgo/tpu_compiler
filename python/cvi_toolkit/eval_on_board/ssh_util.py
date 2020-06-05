#!/usr/bin/env python3

import numpy as np
import sys
import os

def ssh_handshake(args, x):
    input_name = "in.npz"
    out_name = "out.npz"
    np.savez(input_name, input=x)

    #print("input_name:", input_name, md5(input_name))
    cmd = "{} --input {} --model {} --output {}".format(
            args.model_runner_path, input_name, args.model, out_name)

    # send input to board
    _cmd = "sshpass -p '{}' scp -r {} {}@{}:{}".format(
            args.passwd, input_name, args.user, args.board_ip, args.board_path)

    if os.system(_cmd) != 0:
        print('Cmd {} execute failed'.format(_cmd))
        exit(-1)

    # do inference
    cmd = "sshpass -p '{}' ssh -t {}@{} 'cd {} && {}'".format(
            args.passwd, args.user, args.board_ip, args.board_path, cmd)

    if os.system(cmd) != 0:
        print('Cmd {} execute failed'.format(cmd))
        exit(-1)

    # get result
    _cmd = "sshpass -p '{}' scp -r {}@{}:{}/{} . ".format(
            args.passwd, args.user, args.board_ip, args.board_path, out_name)

    if os.system(_cmd) != 0:
        print('Cmd {} execute failed'.format(_cmd))
        exit(-1)

    return out_name


