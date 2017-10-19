#!/usr/bin/env python2.7

from __future__ import print_function

import os, sys, numpy as np
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil

parser = argparse.ArgumentParser()
parser.add_argument('caffemodel', help='path to model')
parser.add_argument('deployproto', help='path to deploy prototxt template')
parser.add_argument('img0', help='image 0 path')
parser.add_argument('img1', help='image 1 path')
parser.add_argument('out',  help='output filename')
parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')

args = parser.parse_args()

if(not os.path.exists(args.caffemodel)): raise BaseException('caffemodel does not exist: '+args.caffemodel)
if(not os.path.exists(args.deployproto)): raise BaseException('deploy-proto does not exist: '+args.deployproto)
if(not os.path.exists(args.img0)): raise BaseException('img0 does not exist: '+args.img0)
if(not os.path.exists(args.img1)): raise BaseException('img1 does not exist: '+args.img1)

num_blobs = 2
input_data = []
img0 = misc.imread(args.img0)
if len(img0.shape) < 3: input_data.append(img0[np.newaxis, np.newaxis, :, :])
else:                   input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
img1 = misc.imread(args.img1)
if len(img1.shape) < 3: input_data.append(img1[np.newaxis, np.newaxis, :, :])
else:                   input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])

width = input_data[0].shape[3]
height = input_data[0].shape[2]
vars = {}
vars['TARGET_WIDTH'] = width
vars['TARGET_HEIGHT'] = height

divisor = 64.
vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);

tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)

proto = open(args.deployproto).readlines()
for line in proto:
    for key, value in vars.items():
        tag = "$%s$" % key
        line = line.replace(tag, str(value))

    tmp.write(line)

tmp.flush()

if not args.verbose:
    caffe.set_logging_disabled()
caffe.set_device(args.gpu)
caffe.set_mode_gpu()
net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)

input_dict = {}
for blob_idx in range(num_blobs):
    input_dict[net.inputs[blob_idx]] = input_data[blob_idx]


net.forward(**input_dict)

total_macs = 0
for name, layer in zip(net._layer_names, net.layers):
    if ((layer.type == 'Convolution') or (layer.type == 'Deconvolution')):
        print(name)
        top_name = net.top_names[name][0]
        top_blob = net.blobs[top_name]
        weights  = net.params[name][0]

        top_blob_shape = top_blob.shape
        layer_height   = top_blob_shape[2]
        layer_width    = top_blob_shape[3]
        print(layer_width)
        print(layer_height)

        weight_shape  = weights.shape
        in_chan       = weight_shape[0]
        out_chan      = weight_shape[1]
        kernel_height = weight_shape[2]
        kernel_width  = weight_shape[3]
        print(in_chan)
        print(out_chan)
        print(kernel_height)
        print(kernel_width)

        # Compute number of multiply accumulate operations
        num_outputs      = layer_width * layer_height * out_chan
        num_macs_per_out = in_chan * kernel_height * kernel_width
        num_layer_macs   = num_outputs * num_macs_per_out
        print(num_layer_macs)

        total_macs      += num_layer_macs


print('===========')
print('Total MACs: '+str(total_macs))



