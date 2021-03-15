import onnx
from onnx import numpy_helper
import sys
import numpy as np

def statistic(array, name):
    res = {
            #'value': array,
            'min_abs': abs(array).min(),
            'max_abs': abs(array).max(),
            'mean_abs': abs(array).mean(),
            'std_abs': abs(array).std(),
            }
    return {name + '_' + n: v for n, v in res.items()}
    
model = onnx.load(sys.argv[1])
# Find target bn
init = model.graph.initializer
weights = dict()
for i in init:
    weights[i.name] = numpy_helper.to_array(i)

stat = dict()
for node in model.graph.node:
    if 'batch' in node.name.lower() and \
    0.0 < abs(weights[node.input[3]]).min() < 1e-32:
        stat_layer = dict()
        # Save FQ, BN and conv weights
        # 1. BN
        denorm_ch_idx = set()
        for w_name in node.input[1:]:
            # 1.5 Find denormal channels
            denorm_ch_idx |= set(np.where(abs(weights[w_name]) < 1e-32)[0].tolist())

        denorm_ch_idx = list(denorm_ch_idx)
        for idx in denorm_ch_idx:
            for w_name in node.input[1:]:
                name = '_ch_' + str(idx)
                if name not in stat_layer:
                    stat_layer[name] = dict()
                stat_layer[name][w_name] = weights[w_name][idx]
        # 2. FQ 
        # Find conv
        for n in model.graph.node:
            if n.output[0] == node.input[0]:
                # n is conv
                # now find weights FQ
                for fq in model.graph.node:
                    if fq.output[0] == n.input[1]:
                        # save all inputs of FQ
                        weights_fq = dict()
                        conv_name_w = fq.input[0]
                        weights_fq['conv_weights'] = weights[conv_name_w]
                        #stat_layer['conv_name'] = conv_name_w
                        for i, inp in enumerate(fq.input[1:]):
                            for const in model.graph.node:
                                if const.output[0] == inp:
                                    weights_fq['fq_inp_' + str(i + 1)] = numpy_helper.to_array(const.attribute[0].t)
        # Save it for each channel
        for idx in denorm_ch_idx:
            for w_name, w_value in weights_fq.items():
                name = '_ch_' + str(idx)
                stat_layer[name][w_name] = w_value[idx]
                if w_name == 'conv_weights':
                    stat_layer[name].update(statistic(w_value[idx], w_name))
        stat[node.name] = stat_layer


import pandas as pd
import sys
import numpy
#numpy.set_printoptions(threshold=sys.maxsize)

res = dict()
for layer, layer_v in stat.items():
    for w_name, w_w in layer_v.items():
        val = {k.split('.')[-1]: v for k, v in w_w.items()}
        res[layer + w_name] = val

with open('denorm.csv', 'w') as out:
    out.write(pd.DataFrame(res).T.to_csv())


        
