
# import pylearn2
# import numpy
import cPickle as pickle
import gzip

# reads the train data saved by pylearn2 and dumps it into a pickled file
def save_train_data():
    with open('/home/siddharth/pylearn2/data/cifar10/pylearn2_gcn_whitened/train.pkl','rb') as fp:
        train_data = pickle.load(fp)
        train_X = train_data.X
        print 'train_X: ', train_X.shape
        train_y = train_data.y
        print 'train_y: ', train_y.shape
        train_data_final = (train_X, train_y)
        with gzip.open('/home/siddharth/pylearn2/data/cifar10/pylearn2_gcn_whitened/train_whitened_gcn.pklz', 'wb') as fp:
            pickle.dump(train_data_final, fp, protocol=2)

# reads the train data saved by pylearn2 and dumps it into a pickled file
def save_test_data():
    with open('/home/siddharth/pylearn2/data/cifar10/pylearn2_gcn_whitened/test.pkl','rb') as fp:
        test_data = pickle.load(fp)
        test_X = test_data.X
        print 'test_X: ', test_X.shape
        test_y = test_data.y
        print 'test_y: ', test_y.shape
        test_data_final = (test_X, test_y)
        with gzip.open('/home/siddharth/pylearn2/data/cifar10/pylearn2_gcn_whitened/test_whitened_gcn.pklz', 'wb') as fp:
            pickle.dump(test_data_final, fp, protocol=2)
            
if __name__ == '__main__':
    save_train_data()
    save_test_data()