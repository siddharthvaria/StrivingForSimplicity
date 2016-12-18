
import numpy
import sys
import os
import timeit
import theano
import theano.tensor as T
import gzip
import cPickle

from utils import load_data2
from Models import Large_AllCNN, Large_AllCNN_BN
from utils import augmentImages, enlargeMiniBatch

def test_Large_AllCNN_Model(use_bn=False, learning_rate=0.05, n_epochs=350, batch_size=200, L2_reg=0.001, 
                            input_ndo_p=0.8, layer_ndo_p=0.5, lrelu_alpha=0.181, save_model=True, save_freq=50, 
                            s1=5, s2=5):
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

#     datasets = load_data2()
# 
#     train_set_x, train_set_y = datasets[0]
#     valid_set_x, valid_set_y = datasets[1]
#     test_set_x, test_set_y = datasets[2]
# 
#     # compute number of minibatches for training, validation and testing
#     n_train_batches = train_set_x.get_value(borrow=True).shape[0]
#     n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
#     n_test_batches = test_set_x.get_value(borrow=True).shape[0]
#     
#     
#     n_train_batches //= batch_size
#     n_valid_batches //= batch_size
#     n_test_batches //= batch_size
# 
#     print 'n_train_batches: ', n_train_batches
#     print 'n_valid_batches: ', n_valid_batches
#     print 'n_test_batches: ', n_test_batches
#     
#     learning_rate = numpy.asarray(learning_rate, dtype=numpy.float32)
#     print 'learning_rate: ', learning_rate

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
    #dropout_input = T.switch(T.neq(training_enabled, 0), drop(x, p=input_ndo_p), input_ndo_p * x)

        
    ##input of 126x126 with color and spatial augmentation and no dropout

#     layer0 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=dropout_input, ## extreme augmented data input
#     filter_shape=(320, 3, 2, 2),
#     image_shape=(batch_size, 3, 126, 126),
#     ssample=(1,1),
#     bordermode='valid',
#     p=1.0,
#     alpha=0.5  ##leaky relu
#     )
# 
#     layer1 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer0.output,
#     filter_shape=(320, 320, 2, 2),
#     image_shape=(batch_size, 320, 125, 125),
#     ssample=(1,1),
#     bordermode='valid',
#     p=1.0,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer2 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer1.output,
#     filter_shape=(320, 320, 2, 2),
#     image_shape=(batch_size, 320, 124, 124),
#     ssample=(2,2),
#     bordermode='valid',
#     p=1.0,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer3 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer2.output,
#     filter_shape=(640, 320, 2, 2),
#     image_shape=(batch_size, 320, 62, 62),
#     ssample=(1,1),
#     bordermode='valid',
#     p=0.9,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer4 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer3.output,
#     filter_shape=(640, 640, 2, 2),
#     image_shape=(batch_size, 640, 61, 61),
#     ssample=(1,1),
#     bordermode='valid',
#     p=0.9,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer5 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer4.output,
#     filter_shape=(640, 640, 2, 2),
#     image_shape=(batch_size, 640, 60, 60),
#     ssample=(2,2),
#     bordermode='valid',
#     p=1.0,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer6 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer5.output,
#     filter_shape=(960, 640, 2, 2),
#     image_shape=(batch_size, 640, 30, 30),
#     ssample=(1,1),
#     bordermode='valid',
#     p=0.8,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer7 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer6.output,
#     filter_shape=(960, 960, 2, 2),
#     image_shape=(batch_size, 960, 29, 29),
#     ssample=(1,1),
#     bordermode='valid',
#     p=0.8,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer8 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer7.output,
#     filter_shape=(960, 960, 2, 2),
#     image_shape=(batch_size, 960, 28, 28),
#     ssample=(2,2),
#     bordermode='valid',
#     p=1.0,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer9 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer8.output,
#     filter_shape=(1280, 960, 2, 2),
#     image_shape=(batch_size, 960, 14, 14),
#     ssample=(1,1),
#     bordermode='valid',
#     p=0.7,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer10 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer9.output,
#     filter_shape=(1280, 1280, 2, 2),
#     image_shape=(batch_size, 1280, 13, 13),
#     ssample=(1,1),
#     bordermode='valid',
#     p=0.7,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer11 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer10.output,
#     filter_shape=(1280, 1280, 2, 2),
#     image_shape=(batch_size, 1280, 12, 12),
#     ssample=(2,2),
#     bordermode='valid',
#     p=1.0,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer12 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer11.output,
#     filter_shape=(1600, 1280, 2, 2),
#     image_shape=(batch_size, 1280, 6, 6),
#     ssample=(1,1),
#     bordermode='valid',
#     p=0.6,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer13 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer12.output,
#     filter_shape=(1600, 1600, 2, 2),
#     image_shape=(batch_size, 1600, 5, 5),
#     ssample=(1,1),
#     bordermode='valid',
#     p=0.6,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer14 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer13.output,
#     filter_shape=(1600, 1600, 2, 2),
#     image_shape=(batch_size, 1600, 4, 4),
#     ssample=(2,2),
#     bordermode='valid',
#     p=1.0,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer15 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer14.output,
#     filter_shape=(1920, 1600, 2, 2),
#     image_shape=(batch_size, 1600, 2, 2),
#     ssample=(1,1)
#     bordermode='valid',
#     p=0.5,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer16 = myConvLayer(
#     rng,
#     is_train=training_enabled,
#     input_data=layer15.output,
#     filter_shape=(1920, 1920, 1, 1),
#     image_shape=(batch_size, 1920, 1, 1),
#     ssample=(1,1),
#     bordermode='valid',
#     p=0.5,
#     alpha=0.5  ##leaky relu
# )
# 
#     layer17 = myConvLayer(
#         rng,
#         is_train=training_enabled,
#         input_data=layer16.output,
#         filter_shape=(10,1920,1,1),
#         image_shape=(batch_size,1920,1,1),
#         ssample=(1,1),
#         bordermode='valid',
#         p=1.0,
#         alpha=0.5 ##leaky relu
# 
#     )
# 
# # make sure this is what global averaging does
# ## no global_average=layer8.output.mean(axis=(2,3))
# ## directly softmax layer
# 
#     # make sure this is what global averaging does
#     # global_average=layer8.output.mean(axis=(2,3))
# 
#     softmax_layer=SoftmaxWrapper(input_data=layer17.output, n_in=10, n_out=10)
# 
#     L2_sqr = (
#                 (layer0.W ** 2).sum()
#                 +(layer1.W**2).sum()
#                 +(layer2.W**2).sum()
#                 +(layer3.W**2).sum()
#                 +(layer4.W**2).sum()
#                 +(layer5.W**2).sum()
#                 +(layer6.W**2).sum()
#                 +(layer7.W**2).sum()
#                 +(layer8.W**2).sum()
#                 +(layer9.W**2).sum()
#                 +(layer10.W**2).sum()
#                 +(layer11.W**2).sum()
#                 +(layer12.W**2).sum()
#                 +(layer13.W**2).sum()
#                 +(layer14.W**2).sum()
#                 +(layer15.W**2).sum()
#                 +(layer16.W**2).sum()
#                 +(layer17.W**2).sum()
# 
#     )
# 
#     # the cost we minimize during training is the NLL of the model
#     cost = (softmax_layer.negative_log_likelihood(y) + L2_reg*L2_sqr)

    classifier = None
    
    if use_bn:
        classifier = Large_AllCNN_BN(rng, 
                           x, 
                           y,
                           batch_size, 
                           training_enabled, 
                           layer_ndo_p, 
                           L2_reg,
                           lrelu_alpha
                           )
    else:
        classifier = Large_AllCNN(rng, 
                           x, 
                           y,
                           batch_size, 
                           training_enabled, 
                           layer_ndo_p, 
                           L2_reg,
                           lrelu_alpha
                           )


    # create a function to compute the mistakes that are made by the model
#     test_model = theano.function(
#         [index],
#         classifier.errors,
#         givens={
#             x: test_set_x[index * batch_size: (index + 1) * batch_size],
#             y: test_set_y[index * batch_size: (index + 1) * batch_size],
#             training_enabled: numpy.cast['int32'](0)
#         }
#     )

    test_model = theano.function(
        [x, y],
        classifier.errors,
        givens={
            training_enabled: numpy.cast['int32'](0)
        }
    )

#     validate_model = theano.function(
#         [index],
#         classifier.errors,
#         givens={
#             x: valid_set_x[index * batch_size: (index + 1) * batch_size],
#             y: valid_set_y[index * batch_size: (index + 1) * batch_size],
#             training_enabled: numpy.cast['int32'](0)
#         }
#     )

    validate_model = theano.function(
        [x, y],
        classifier.errors,
        givens={
            training_enabled: numpy.cast['int32'](0)
        }
    )
    
    # create a list of all model parameters to be fit by gradient descent
    #params = layer17.params + layer16.params + layer15.params + layer14.params + layer13.params + layer12.params + layer11.params + layer10.params + layer9.params + layer8.params + layer7.params + layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.

    momentum =theano.shared(numpy.cast[theano.config.floatX](0.9), name='momentum')
    updates = []
    for param in classifier.params:
        param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))    
        updates.append((param, param - lr * param_update))
        updates.append((param_update, momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*T.grad(classifier.cost, param)))

#     train_model = theano.function(
#         [index, lr],
#         classifier.cost,
#         updates=updates,
#         givens={
#             x: train_set_x[index * batch_size: (index + 1) * batch_size],
#             y: train_set_y[index * batch_size: (index + 1) * batch_size],
#             training_enabled: numpy.cast['int32'](1)
#         }
#     )

    train_model = theano.function(
        [x, y, lr],
        classifier.cost,
        updates=updates,
        givens={
            training_enabled: numpy.cast['int32'](1)
        }
    )

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
    if use_bn:
        updateLRAfter = 100
    else:
        updateLRAfter = 200

    print 'Using Batch Normalization: ', use_bn
    print 'updateLRAfter: ', updateLRAfter
    
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

            train_x = augmentImages(train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size], shift1=s1, shift2=s2, enlarge=True)
            train_y = train_set_y[minibatch_index* batch_size: (minibatch_index + 1) * batch_size]
            cost_ij = train_model(train_x, train_y, learning_rate)
            
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(enlargeMiniBatch(valid_set_x[ii * batch_size: (ii + 1) * batch_size]), valid_set_y[ii * batch_size: (ii + 1) * batch_size]) for ii
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
                        test_model(enlargeMiniBatch(test_set_x[ii * batch_size: (ii + 1) * batch_size]), test_set_y[ii * batch_size: (ii + 1) * batch_size])
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
                cPickle.dump([param.get_value() for param in classifier.params], fp, protocol=2)
        
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
    test_Large_AllCNN_Model(use_bn=False, n_epochs=350)
    #test_Large_AllCNN_Model(use_bn=True, n_epochs=150)
    