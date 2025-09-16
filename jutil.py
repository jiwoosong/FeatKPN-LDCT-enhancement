import sys, os
import termcolor
import numpy as np
import time

'''Terminal'''
# convert to colored strings
def toWhite(content): return termcolor.colored(content,"white",attrs=["bold"])
def toRed(content): return termcolor.colored(content,"red",attrs=["bold"])
def toGreen(content): return termcolor.colored(content,"green",attrs=["bold"])
def toBlue(content): return termcolor.colored(content,"blue",attrs=["bold"])
def toCyan(content): return termcolor.colored(content,"cyan",attrs=["bold"])
def toYellow(content): return termcolor.colored(content,"yellow",attrs=["bold"])
def toMagenta(content): return termcolor.colored(content,"magenta",attrs=["bold"])
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    '''
    Usage:
    for i in range(100):
        printProgress(i+1, 99, 'Progress:', '', 1, 50)
    '''
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    if iteration == total:
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix + toGreen('\tDone.')))
        sys.stdout.write('\n')
    else:
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix))
    sys.stdout.flush()
def str_100(msg):
    msglen = len(msg)
    if len(msg)>100:
        return('--'+msg+'--')
    else:
        a = (100-msglen)/2
        return('-'*int(np.floor(a))+msg+'-'*int(np.ceil(a)))
'''Model'''
def load_checkpoint(net, weight_param, key='', p=True):
    '''
    :param net: Network
    :param param: Weight params
    :param p: print option
    :return: Network
    '''
    model_dict = net.state_dict()
    for name, param in weight_param.items():
        if name.__contains__(key):
            if key != '':
                name = '.'.join(name.split('.')[1:])
            if name not in model_dict:
                if p:
                    print(name + 'Not Loaded')
                continue
            else:
                if model_dict[name].shape == param.shape:
                    model_dict[name].copy_(param)
                else:
                    print(name + 'Not Loaded (shape difference)')
    return net
'''Util'''
def listdir(p):
    return [os.path.join(p,item) for item in os.listdir(p)]