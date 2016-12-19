import numpy
import sys
import os
import timeit
import theano
import theano.tensor as T
import gzip
import cPickle
from Models import ModelA_AllCNN, ModelB_AllCNN, ModelC_AllCNN, ModelA_AllCNN_BN, ModelB_AllCNN_BN, ModelC_AllCNN_BN
from utils import augmentImages

from utils import load_data2, drop

def test_AllCNN_Models_DA_BN(use_bn=False, model='c', learning_rate=0.05, n_epochs=350, batch_size=200, L2_reg=0.001, input_ndo_p=0.8, layer_ndo_p=0.5, save_model=True, save_freq=50, s1=5, s2=5):
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

    classifier = None
    Model_Name = None
    if use_bn:
        if model == 'a':
            Model_Name = ModelA_AllCNN_BN
        elif model == 'b':
            Model_Name = ModelB_AllCNN_BN
        elif model == 'c':
            Model_Name = ModelC_AllCNN_BN
        else:
            raise RuntimeError('Invalid model parameter!')
    else:
        if model == 'a':
            Model_Name = ModelA_AllCNN
        elif model == 'b':
            Model_Name = ModelB_AllCNN
        elif model == 'c':
            Model_Name = ModelC_AllCNN
        else:
            raise RuntimeError('Invalid model parameter!')

    classifier = Model_Name(rng, 
                           dropout_input, 
                           y, 
                           batch_size, 
                           training_enabled, 
                           layer_ndo_p, 
                           L2_reg
                           )
        
    print 'Training Model: ', classifier.__class__.__name__
    
    test_model = theano.function(
        [x, y],
        classifier.errors,
        givens={
            training_enabled: numpy.cast['int32'](0)
        }
    )
    
    validate_model = theano.function(
        [x, y],
        classifier.errors,
        givens={
            training_enabled: numpy.cast['int32'](0)
        }
    )

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.

    momentum =theano.shared(numpy.cast[theano.config.floatX](0.9), name='momentum')
    updates = []
    for param in  classifier.params:
        param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))    
        updates.append((param, param - lr * param_update))
        updates.append((param_update, momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*T.grad(classifier.cost, param)))

    
    train_model = theano.function(
        [x, y, lr],
        classifier.cost,
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
    if use_bn:
        updateLRAfter = 100
    else:
        updateLRAfter = 200    

    while (epoch < n_epochs) and (not done_looping):
        
        # shuffle data before starting the epoch
        
        epoch = epoch + 1
        if(epoch > updateLRAfter):
            learning_rate *= 0.1
            updateLRAfter += 50

        for minibatch_index in range(n_train_batches):
            #print 'epoch: {0}, minibatch: {1}'.format(epoch, minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
#             if iter % 50 == 0:
#                 print('training @ iter = ', iter)

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
                cPickle.dump([param.get_value() for param in classifier.params], fp, protocol=2)
        
    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
#     print(('The code for file ' +
#            os.path.split(__file__)[1] +
#            ' ran for %.2fm' % ((end_time - start_time) / 60.)), sys.stderr)

if __name__ == '__main__':
    test_AllCNN_Models_DA_BN(use_bn=True, model='c')