from __future__ import unicode_literals
import sys
import os

if __name__ == '__main__':
    assert(len(sys.argv)==3), \
        ["Usage: python get_flo_for_keyframes.py [DEVKIT_PATH] [TXT_PATH]"]

    devkit_path = sys.argv[1]+'/'
    txt_path    = sys.argv[2]+'/'

    datasets      = ['yt_bb_detection_train',
                     'yt_bb_detection_validation']

    flo_command   = './run-network.sh -n FlowNet2-s -g 1 -vv'

    for d_set in datasets:
        with open(txt_path+d_set+'_key_frame_file_list.txt') as f:
            key_frame_file_list = f.read().split("\n")
        with open(txt_path+d_set+'_inf_frame_file_list.txt') as f:
            inf_frame_file_list = f.read().split("\n")
        with open(txt_path+d_set+'_flow_data_file_list.txt') as f:
            flow_data_file_list = f.read().split("\n")

        for idx in xrange(len(key_frame_file_list)):
            key_list = key_frame_file_list[idx]
            inf_list = inf_frame_file_list[idx]
            flo_list = flow_data_file_list[idx]

            command = flo_command + ' ' + \
                      key_list + ' ' + \
                      inf_list + ' ' + \
                      flo_list

            os.system(command)
