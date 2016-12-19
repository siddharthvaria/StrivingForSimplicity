import gzip
import cPickle
import numpy as np
from matplotlib import pyplot as plt

weights_file = '/home/siddharth/workspace/StrivingForSimplicity/params/parameters_epoch_100.pklz'

def plot_feature_maps(weights_file):
        with gzip.open(weights_file, 'rb') as fp:                
            #cPickle.dump([param.get_value() for param in classifier.params], fp, protocol=2)
            parameters = cPickle.load(fp)
            parameters = np.asarray(parameters)
            plot_single_layer(32, 0, parameters)
            plot_single_layer(28, 0, parameters)
            
def plot_single_layer(param_num, feature_map_num, layer_params):
    # extract the params of given layer_num and feature_map_num
    params = layer_params[param_num]
    params = params[:,feature_map_num,:,:]
    print 'params.shape: ', params.shape
    f, axarr = plt.subplots(12,8)
    c = 0
    for i in range(12):
        for j in range(8):
            plt.axes(axarr[i,j])
            plt.axis('off')
            plt.imshow(params[c], cmap='Greys')
            c += 1
    f.savefig('/home/siddharth/workspace/StrivingForSimplicity/params/params_{0}_{1}.png'.format(param_num, feature_map_num))
    plt.close(f)

if __name__ == '__main__':
    plot_feature_maps(weights_file)