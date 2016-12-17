import numpy
import sys
import os
import timeit
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import batch_normalization
import gzip
import cPickle
import random
from scipy import ndimage
from matplotlib import pyplot as plt
#from skimage import transform

from utils import load_data1, load_data2, drop

# def translateMiniBatch(total_images, x, y):
#     if x == 0 and y == 0:
#         return total_images
#     nImg = numpy.zeros((len(total_images),3,32,32))
#     ii = 0
#     for img in total_images:
#         cFlip = random.randint(0,1)
#         if(cFlip):
#             shift1 = random.randint(-x,x)
#             shift2 = random.randint(-y,y)
#             tf = transform.AffineTransform(translation=(shift1,shift2))
#         else:
#             tf = transform.AffineTransform(scale=(1,1))
# 
#         nImg[ii][0] = transform.warp(img[0], tf)
#         nImg[ii][1] = transform.warp(img[1], tf)
#         nImg[ii][2] = transform.warp(img[2], tf)        
#         ii+=1
#     return numpy.asarray(nImg, dtype=theano.config.floatX)

def augmentImages(X, shift1=0, shift2=0):
    X_augmented = []
    for x in X:
        isAugment = random.randint(0,1)
        if isAugment:
            # randomly decide to flip the image or translate
            cFlip = random.randint(0,1)
            if cFlip:
                # flip the image
                X_augmented.append(flipSingleImage(x))
            else:
                # translate the image
                X_augmented.append(translate_single_image(x, numpy.random.random_integers(0,shift1), numpy.random.random_integers(0,shift2)))
        else:
            X_augmented.append(x)
    
    X_augmented = numpy.asarray(X_augmented, dtype=theano.config.floatX)
    assert (X.shape == X_augmented.shape), 'Dimension mismatch while augmenting images!'
    return X_augmented

def translate_single_image(img,x_shift,y_shift):
    return [ndimage.interpolation.shift(x,(x_shift,y_shift)) for x in img]

# def translate_image(X, shift1, shift2):
#     X_translated = [translate_single_image(X[ii], shift1, shift2) for ii in xrange(len(X))]
#     return numpy.asarray(X_translated, dtype=theano.config.floatX)

def flipSingleImage(img):
    return [numpy.fliplr(x) for x in img]

# def flipSingleImage(img):
#     assert (img.shape[1] == img.shape[2]), 'This method Only works for symmetric images!'
#     size = img.shape[1]
#     nimg = numpy.zeros((img.shape[0],size,size))
#     for ii in xrange(img.shape[0]):
#         for y in range(size):
#             for x in range(size//2):
#                 left = img[ii][x][y]
#                 right = img[ii][size - 1 - x][y]
#                 nimg[ii][size - 1 - x][y] = left
#                 nimg[ii][x][y]= right
#     return nimg

def plotImages(rseed, shift1=0, shift2=0, aug=True):
    datasets = load_data2(theano_shared=False)
    test_set_x, test_set_y = datasets[2]
    X = test_set_x[0:501].reshape(501,3,32,32)
    if aug:
        X_new = augmentImages(X, shift1=shift1, shift2=shift2)
    else:
        X_new = X
    f, axarr = plt.subplots(4,4)
    c = 0
    for i in range(4):
        for j in range(4):
            plt.axes(axarr[i,j])
            plt.imshow(X_new[rseed[c]].transpose(1,2,0))
            c += 1
    f.savefig('/home/siddharth/workspace/StrivingForSimplicity/augmented_images_{0}.png'.format(int(aug)))
    plt.close(f)

# def flipMiniBatch(total_images):
#     total_images_flipped = [flipSingleImage(x) for x in total_images]
#     return numpy.asarray(total_images_flipped, dtype=theano.config.floatX)

class SoftmaxWrapper(object):
    def __init__(self, input_data, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        
        self.p_y_given_x = T.nnet.softmax(input_data)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # keep track of model input
        self.input = input_data

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class myConvLayerBN(object):

    def __init__(self, rng, is_train, input_data, filter_shape, image_shape, ssample=(1,1), bordermode='valid', p=0.5):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        
        assert image_shape[1] == filter_shape[1]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(ssample))
        
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        gamma_values = numpy.ones((filter_shape[0],), dtype=theano.config.floatX)
        self.gamma = theano.shared(value = gamma_values, borrow=True)
        
        beta_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.beta = theano.shared(value = beta_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input_data,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            subsample=ssample,
            border_mode=bordermode
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        
        lin_output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        
        bn_output = batch_normalization(inputs = lin_output,
            gamma = self.gamma.dimshuffle('x',0,'x','x'), beta = self.beta.dimshuffle('x',0,'x','x'), mean = lin_output.mean((0,), keepdims=True),
            std = lin_output.std((0,), keepdims = True),
                        mode='low_mem')
        
        activated_output = T.nnet.relu(bn_output)
        
        dropped_output = drop(activated_output,p)
        
        self.output = T.switch(T.neq(is_train, 0), dropped_output, p*activated_output)
        
        # store parameters of this layer
        self.params = [self.W, self.b, self.gamma, self.beta]

        # keep track of model input
        self.input = input_data

def ModelC_AllCNN_BN(learning_rate=0.05, n_epochs=350, batch_size=200, L2_reg=0.001, input_ndo_p=0.8, layer_ndo_p=0.5, save_model=True, save_freq=50, s1=5, s2=5):
    """
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    
    :type batch_size: int
    :param batch_size: the number of training examples per batch
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data2(theano_shared=False)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    train_set_x = train_set_x.reshape(len(train_set_x),3,32,32)
    valid_set_x = valid_set_x.reshape(len(valid_set_x),3,32,32)
    test_set_x = test_set_x.reshape(len(test_set_x),3,32,32)
    
    train_set_x = numpy.asarray(train_set_x, dtype=theano.config.floatX)
    valid_set_x = numpy.asarray(valid_set_x, dtype=theano.config.floatX)
    test_set_x = numpy.asarray(test_set_x, dtype=theano.config.floatX)
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0]
    n_valid_batches = valid_set_x.shape[0]
    n_test_batches = test_set_x.shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    print 'n_train_batches: ', n_train_batches
    print 'n_valid_batches: ', n_valid_batches
    print 'n_test_batches: ', n_test_batches
    
    learning_rate = numpy.asarray(learning_rate, dtype=numpy.float32)
    print 'learning_rate: ', learning_rate

    # allocate symbolic variables for the data
    #index = T.lscalar()  # index to a [mini]batch
    lr = T.fscalar()
    training_enabled = T.iscalar('training_enabled')

    # start-snippet-1
    x = T.tensor4('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    #layer0_input = x.reshape((batch_size, 3, 32, 32))

    # drop the input only while training, don't drop while testing
    #dropout_input = T.switch(T.neq(training_enabled, 0), drop(layer0_input, p=input_ndo_p), input_ndo_p * layer0_input)
    dropout_input = T.switch(T.neq(training_enabled, 0), drop(x, p=input_ndo_p), input_ndo_p * x)

    layer0 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=dropout_input,
        filter_shape=(96, 3, 3, 3),
        image_shape=(batch_size, 3, 32, 32),
        ssample=(1,1),
        bordermode='half',
        p=1.0
    )

    layer1 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer0.output,
        filter_shape=(96, 96, 3, 3),
        image_shape=(batch_size, 96, 32, 32),
        ssample=(1,1),
        bordermode='half',
        p=1.0
    )

    layer2 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer1.output,
        filter_shape=(96, 96, 3, 3),
        image_shape=(batch_size, 96, 32, 32),
        ssample=(2,2),
        bordermode='half',
        p=layer_ndo_p
    )

    layer3 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer2.output,
        filter_shape=(192, 96, 3, 3),
        image_shape=(batch_size, 96, 16, 16),
        ssample=(1,1),
        bordermode='half',
        p=1.0
    )

    layer4 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer3.output,
        filter_shape=(192, 192, 3, 3),
        image_shape=(batch_size, 192, 16, 16),
        ssample=(1,1),
        bordermode='half',
        p=1.0
    )

    layer5 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer4.output,
        filter_shape=(192, 192, 3, 3),
        image_shape=(batch_size, 192, 16, 16),
        ssample=(2,2),
        bordermode='half',
        p=layer_ndo_p
    )

    layer6 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer5.output,
        filter_shape=(192, 192, 3, 3),
        image_shape=(batch_size, 192, 8, 8),
        ssample=(1,1),
        bordermode='half',
        p=1.0
    )

    layer7 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer6.output,
        filter_shape=(192, 192, 1, 1),
        image_shape=(batch_size, 192, 8, 8),
        ssample=(1,1),
        bordermode='half',
        p=1.0
    )

    layer8 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer7.output,
        filter_shape=(10, 192, 1, 1),
        image_shape=(batch_size, 192, 8, 8),
        ssample=(1,1),
        bordermode='half',
        p=1.0
    )

    # make sure this is what global averaging does
    global_average=layer8.output.mean(axis=(2,3))

    softmax_layer=SoftmaxWrapper(input_data=global_average, n_in=10, n_out=10)

    L2_sqr = (
                (layer0.W ** 2).sum()
                +(layer1.W**2).sum()
                +(layer2.W**2).sum()
                +(layer3.W**2).sum()
                +(layer4.W**2).sum()
                +(layer5.W**2).sum()
                +(layer6.W**2).sum()
                +(layer7.W**2).sum()
                +(layer8.W**2).sum()
    )

    # the cost we minimize during training is the NLL of the model
    cost = (softmax_layer.negative_log_likelihood(y) + L2_reg*L2_sqr)

    # create a function to compute the mistakes that are made by the model
#     test_model = theano.function(
#         [index],
#         softmax_layer.errors(y),
#         givens={
#             x: test_set_x[index * batch_size: (index + 1) * batch_size],
#             y: test_set_y[index * batch_size: (index + 1) * batch_size],
#             training_enabled: numpy.cast['int32'](0)
#         }
#     )
    
    test_model = theano.function(
        [x, y],
        softmax_layer.errors(y),
        givens={
            training_enabled: numpy.cast['int32'](0)
        }
    )

#     validate_model = theano.function(
#         [index],
#         softmax_layer.errors(y),
#         givens={
#             x: valid_set_x[index * batch_size: (index + 1) * batch_size],
#             y: valid_set_y[index * batch_size: (index + 1) * batch_size],
#             training_enabled: numpy.cast['int32'](0)
#         }
#     )
    
    validate_model = theano.function(
        [x, y],
        softmax_layer.errors(y),
        givens={
            training_enabled: numpy.cast['int32'](0)
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer8.params + layer7.params + layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.

    momentum =theano.shared(numpy.cast[theano.config.floatX](0.9), name='momentum')
    updates = []
    for param in  params:
        param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))    
        updates.append((param, param - lr * param_update))
        updates.append((param_update, momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*T.grad(cost, param)))

#     train_model = theano.function(
#         [index, lr],
#         cost,
#         updates=updates,
#         givens={
#             x: train_set_x[index * batch_size: (index + 1) * batch_size],
#             y: train_set_y[index * batch_size: (index + 1) * batch_size],
#             training_enabled: numpy.cast['int32'](1)
#         }
#     )
    
    train_model = theano.function(
        [x, y, lr],
        cost,
        updates=updates,
        givens={
            training_enabled: numpy.cast['int32'](1)
        }
    )
        
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
#     patience = 10000  # look as this many examples regardless
#     patience_increase = 2  # wait this much longer when a new best is found
    
#     improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    
#    validation_frequency = min(n_train_batches, patience // 2)
    
    validation_frequency = n_train_batches // 2
    
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    updateLRAfter = 200

    while (epoch < n_epochs) and (not done_looping):
        
        # shuffle data before starting the epoch
        
        epoch = epoch + 1
        if(epoch > updateLRAfter):
            learning_rate *= 0.1
            updateLRAfter += 50
            print 'epoch: ', epoch
            print 'updateLRAfter: ', updateLRAfter
            print 'learning_rate: ', learning_rate

        for minibatch_index in range(n_train_batches):
            #print 'epoch: {0}, minibatch: {1}'.format(epoch, minibatch_index)
                        
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if iter % 50 == 0:
                print('training @ iter = ', iter)

            train_x = augmentImages(train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size], shift1=s1, shift2=s2)
            train_y = train_set_y[minibatch_index* batch_size: (minibatch_index + 1) * batch_size]
            cost_ij = train_model(train_x, train_y, learning_rate)
            
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(valid_set_x[ii * batch_size: (ii + 1) * batch_size], valid_set_y[ii * batch_size: (ii + 1) * batch_size]) for ii
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
#                     if this_validation_loss < best_validation_loss *  \
#                        improvement_threshold:
#                         patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(test_set_x[ii * batch_size: (ii + 1) * batch_size], test_set_y[ii * batch_size: (ii + 1) * batch_size])
                        for ii in range(n_test_batches)
                    ]
                    
                    test_score= numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

#             if patience <= iter:
#                 done_looping = True
#                 break

        if save_model and epoch % save_freq == 0:
            # add model name to the file to differentiate different models
            with gzip.open('parameters_epoch_{0}.pklz'.format(epoch), 'wb') as fp:                
                cPickle.dump([param.get_value() for param in params], fp, protocol=2)
        
    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), sys.stderr)

if __name__ == '__main__':
#     indices = numpy.random.random_integers(0,high=500,size=(16,))
#     plotImages(indices, 5, 5, aug=True)
#     plotImages(indices, 5, 5, aug=False)
    ModelC_AllCNN_BN()