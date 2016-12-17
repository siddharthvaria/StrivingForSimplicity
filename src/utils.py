from __future__ import print_function
import os
import numpy
import scipy.io
import tarfile
import theano
import theano.tensor as T
from preprocessing import global_contrast_normalize
import cPickle as pickle
import gzip

def drop(input, p=0.5): 
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout is applied
    
    :type p: float or double between 0. and 1. 
    :param p: p probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.
    
    """  
    rng = numpy.random.RandomState(1234)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask

# Below two methods handle dropout of tensors and matrices respectively
# The difference between drop and the below methods is that the below methods
# will drop the same pixel from all the filter maps instead of dropping random
# pixels from each filter map

def drop_tensor(input, p):
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout is applied
    
    :type p: float or double between 0. and 1. 
    :param p: p probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.
    """
    rng = numpy.random.RandomState(1234)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    a,b,c,d = input.shape
    mask = srng.binomial(n=1, p=p, size=(a,c,d), dtype=theano.config.floatX)
    mask = T.tile(mask,(1,b,1)).reshape((a,b,c,d))
    return input * mask

def drop_matrix(input, p):
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout is applied
    
    :type p: float or double between 0. and 1. 
    :param p: p probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.
    """
    rng = numpy.random.RandomState(1234)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    a,b = input.shape
    mask = srng.binomial(n=1, p=p, size=(a,b/3), dtype=theano.config.floatX)
    mask = T.tile(mask,(1,3))
    return input * mask

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_data1(ds_rate=None, theano_shared=True):
    ''' Loads the CIFAR-10 dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    
    if ds_rate is not None:
        assert(ds_rate > 1.)

    # Download the CIFAR-10 dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        #f_name = new_path.replace("src/../data/%s"%dataset, "data/") 
        f_name = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data"
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'https://www.cs.toronto.edu/~kriz/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path) 
             
        tar = tarfile.open(new_path)
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name,f_name)
        tar.close()              
        
        return f_name
    
    f_name=check_dataset('cifar-10-matlab.tar.gz')
    
    # Load data and convert data format
    train_batches=['data_batch_1.mat','data_batch_2.mat','data_batch_3.mat','data_batch_4.mat','data_batch_5.mat']
    train_batch=os.path.join(f_name,'cifar-10-batches-mat',train_batches[0])
    train_set=scipy.io.loadmat(train_batch)
    #train_set['data']=train_set['data']/255.

    # No need to normalize here
    # https://groups.google.com/forum/#!topic/pylearn-users/J-Q-o_FtUl8
    train_set['data']=train_set['data']
    for i in range(4):
        train_batch=os.path.join(f_name,'cifar-10-batches-mat',train_batches[i+1])
        temp=scipy.io.loadmat(train_batch)
        #train_set['data']=numpy.concatenate((train_set['data'],temp['data']/255.),axis=0)
        train_set['data']=numpy.concatenate((train_set['data'],temp['data']),axis=0)
        train_set['labels']=numpy.concatenate((train_set['labels'].flatten(),temp['labels'].flatten()),axis=0)

    
    test_batches=os.path.join(f_name,'cifar-10-batches-mat/test_batch.mat')
    test_set=scipy.io.loadmat(test_batches)
    #test_set['data']=test_set['data']/255.
    test_set['labels']=test_set['labels'].flatten()
    
    # center the data and also contrast normalize the data
    # https://groups.google.com/forum/#!topic/pylearn-users/J-Q-o_FtUl8
    # sqrt_bias=10 for pixel range [0,255]
    # Should we normalize r,g,b components separately ? Not sure

    print ('global contrast normalizing train data ...')
    train_set['data'] = global_contrast_normalize(train_set['data'], subtract_mean=True, use_std=True, sqrt_bias=10)
    print ('global contrast normalizing test data ...')
    test_set['data'] = global_contrast_normalize(test_set['data'], subtract_mean=True, use_std=True, sqrt_bias=10)
    
    train_set=(train_set['data'],train_set['labels'])
    test_set=(test_set['data'],test_set['labels'])
    
    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)
        train_set = [x[:train_set_len] for x in train_set]

    # Extract validation dataset from train dataset
    # valid_set is 1/5th of the train set however this portion is not removed from the train set
    # This is done because train set was not reduced in the paper. 
    valid_set = [x[-(train_set_len//5):] for x in train_set]
    #train_set = [x[:-(train_set_len//5)] for x in train_set]

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval

def load_data2(ds_rate=None, theano_shared=True):
    ''' Loads the CIFAR-10 dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    
    if ds_rate is not None:
        assert(ds_rate > 1.)
        
    train_set = None
    with gzip.open('../data/train_whitened_gcn.pklz','rb') as fp:
        train_set = pickle.load(fp)
    
    train_X, train_y = train_set
    train_y = train_y.flatten()
    
    train_set = (train_X, train_y)

    test_set = None
    with gzip.open('../data/test_whitened_gcn.pklz','rb') as fp:
        test_set = pickle.load(fp)
        
    test_X, test_y = test_set
    test_y = test_y.flatten()
    
    test_set = (test_X, test_y)

    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)
        train_set = [x[:train_set_len] for x in train_set]

    # Extract validation dataset from train dataset
    # valid_set is 1/5th of the train set however this portion is not removed from the train set
    # This is done because train set was not reduced in the paper. 
    valid_set = [x[-(train_set_len//5):] for x in train_set]
    #train_set = [x[:-(train_set_len//5)] for x in train_set]

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval