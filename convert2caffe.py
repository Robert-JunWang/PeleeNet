import sys
import caffe

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
        

def _convert_dense_layer(torch_name, caffe_name):
    print 'converting ', caffe_name
    _convert_basic_block('{}.branch1a'.format(torch_name), '{}/branch1a'.format(caffe_name))
    _convert_basic_block('{}.branch1b'.format(torch_name), '{}/branch1b'.format(caffe_name))

    _convert_basic_block('{}.branch2a'.format(torch_name), '{}/branch2a'.format(caffe_name))
    _convert_basic_block('{}.branch2b'.format(torch_name), '{}/branch2b'.format(caffe_name))
    _convert_basic_block('{}.branch2c'.format(torch_name), '{}/branch2c'.format(caffe_name)) 


def _convert_basic_block(torch_name, caffe_name, torch_bn_postfix='norm'):
    torch_conv = '{}.conv.weight'.format(torch_name)
    caffe_conv = caffe_name
    save_conv2caffe(torch_params[torch_conv].cpu().numpy(), None, params[caffe_conv])

    torch_norm = '{}.{}'.format(torch_name, torch_bn_postfix)
    bn_name = caffe_conv + "/bn"
    scale_name = caffe_conv + "/scale"

    running_mean = torch_params['{}.running_mean'.format(torch_norm)].cpu()
    running_var = torch_params['{}.running_var'.format(torch_norm)].cpu() 
    save_bn2caffe(running_mean, running_var, params[bn_name])
    #print('%s running_mean' % parent_name, running_mean)
    #exit(0)
    scale_weights = torch_params['{}.weight'.format(torch_norm)].cpu()
    scale_biases = torch_params['{}.bias'.format(torch_norm)].cpu()
    save_scale2caffe(scale_weights, scale_biases, params[scale_name])

def _convert_bgr_block(torch_name, caffe_name, torch_bn_postfix='norm'):
    torch_conv = '{}.conv.weight'.format(torch_name)
    caffe_conv = caffe_name

    torch_weight = torch_params[torch_conv].cpu().numpy()
    torch_weight = torch_weight[:, ::-1, ...]

    save_conv2caffe(torch_weight, None, params[caffe_conv])

    torch_norm = '{}.{}'.format(torch_name, torch_bn_postfix)
    bn_name = caffe_conv + "/bn"
    scale_name = caffe_conv + "/scale"

    running_mean = torch_params['{}.running_mean'.format(torch_norm)].cpu()
    running_var = torch_params['{}.running_var'.format(torch_norm)].cpu() 
    save_bn2caffe(running_mean, running_var, params[bn_name])
    #print('%s running_mean' % parent_name, running_mean)
    #exit(0)
    scale_weights = torch_params['{}.weight'.format(torch_norm)].cpu()
    scale_biases = torch_params['{}.bias'.format(torch_norm)].cpu()
    save_scale2caffe(scale_weights, scale_biases, params[scale_name])

def save_conv2caffe(weights, biases, conv_param):
    if biases is not None:
        conv_param[1].data[...] = biases
    conv_param[0].data[...] = weights 

def save_fc2caffe(weights, biases, fc_param):
    fc_param[1].data[...] = biases.numpy() 
    fc_param[0].data[...] = weights.numpy() 

def save_bn2caffe(running_mean, running_var, bn_param):
    bn_param[0].data[...] = running_mean.numpy()
    bn_param[1].data[...] = running_var.numpy()
    bn_param[2].data[...] = np.array([1.0])

def save_scale2caffe(weights, biases, scale_param):
    scale_param[1].data[...] = biases.numpy()
    scale_param[0].data[...] = weights.numpy()

torch_base_name='module.features'


checkpoint = torch.load('weights/peleenet_acc7208.pth.tar')
torch_params = checkpoint['state_dict']

caffemodel='caffe/peleenet.caffemodel'
protofile='caffe/peleenet.prototxt'
net = caffe.Net(protofile, caffe.TEST)
params = net.params


torch_stem_layers = ['module.features.stemblock.stem1',
    'module.features.stemblock.stem2a',
    'module.features.stemblock.stem2b',
    'module.features.stemblock.stem3']
caffe_stem_layers = ['stem1',
    'stem2a',
    'stem2b',
    'stem3']

for i, caffe_conv in enumerate(caffe_stem_layers):
    torch_conv = torch_stem_layers[i]

    if i == 0: # from rgb to BGR
        _convert_bgr_block(torch_conv, caffe_conv)
    else:
        _convert_basic_block(torch_conv, caffe_conv)

# convert dense layers
block_config=(3, 4, 8, 6)
for i, num_layers in enumerate(block_config):
    for k in range(num_layers):
        torch_name = '{}.denseblock{}.denselayer{}'.format(torch_base_name, i + 1, k + 1)
        caffe_name = 'stage{}_{}'.format(i + 1, k + 1)
        _convert_dense_layer(torch_name, caffe_name)

    torch_trans_name = '{}.transition{}'.format(torch_base_name, i + 1)
    _convert_basic_block(torch_trans_name, 'stage{}_tb'.format(i + 1))


# convert classify layer
torch_fc_name = 'module.classifier'
fc_param = params['classifier']
weights = torch_params['{}.weight'.format(torch_fc_name)].cpu()
biases = torch_params['{}.bias'.format(torch_fc_name)].cpu()
save_fc2caffe(weights, biases, fc_param)

print('save caffemodel to %s' % caffemodel)
net.save(caffemodel)
