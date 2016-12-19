from cnn_utils import myConvLayer, myConvLayerBN
from cnn_utils import SoftmaxWrapper

class ModelC_AllCNN(object):
    
    def __init__(self, rng, X_data, y_data, batch_size, training_enabled, layer_ndo_p, L2_reg):
        # border mode: half means conv2d will pad filter_size//2 zeros on all four sides and then use stride=1
        layer0 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=X_data,
            filter_shape=(96, 3, 3, 3),
            image_shape=(batch_size, 3, 32, 32),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer1 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer0.output,
            filter_shape=(96, 96, 3, 3),
            image_shape=(batch_size, 96, 32, 32),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer2 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer1.output,
            filter_shape=(96, 96, 3, 3),
            image_shape=(batch_size, 96, 32, 32),
            ssample=(2,2),
            bordermode='half',
            p=layer_ndo_p
        )
    
        layer3 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer2.output,
            filter_shape=(192, 96, 3, 3),
            image_shape=(batch_size, 96, 16, 16),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer4 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer3.output,
            filter_shape=(192, 192, 3, 3),
            image_shape=(batch_size, 192, 16, 16),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer5 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer4.output,
            filter_shape=(192, 192, 3, 3),
            image_shape=(batch_size, 192, 16, 16),
            ssample=(2,2),
            bordermode='half',
            p=layer_ndo_p
        )
    
        layer6 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer5.output,
            filter_shape=(192, 192, 3, 3),
            image_shape=(batch_size, 192, 8, 8),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer7 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer6.output,
            filter_shape=(192, 192, 1, 1),
            image_shape=(batch_size, 192, 8, 8),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer8 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer7.output,
            filter_shape=(10, 192, 1, 1),
            image_shape=(batch_size, 192, 8, 8),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        global_average=layer8.output.mean(axis=(2,3))
    
        softmax_layer=SoftmaxWrapper(input_data=global_average, n_in=10, n_out=10)
        
        self.errors = softmax_layer.errors(y_data)
    
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
        self.cost = (softmax_layer.negative_log_likelihood(y_data) + L2_reg * L2_sqr)
        
        self.params = layer8.params + layer7.params + layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
        
        self.input = X_data
        self.y = y_data

class ModelB_AllCNN(object):
    def __init__(self, rng, X_data, y_data, batch_size, training_enabled, layer_ndo_p, L2_reg):
        
        layer0 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=X_data,
            filter_shape=(96, 3, 5, 5),
            image_shape=(batch_size, 3, 32, 32),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer1 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer0.output,
            filter_shape=(96, 96, 1, 1),
            image_shape=(batch_size, 96, 32, 32),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer2 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer1.output,
            filter_shape=(96, 96, 3, 3),
            image_shape=(batch_size, 96, 32, 32),
            ssample=(2,2),
            bordermode='half',
            p=layer_ndo_p
        )
    
        layer3 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer2.output,
            filter_shape=(192, 96, 5, 5),
            image_shape=(batch_size, 96, 16, 16),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer4 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer3.output,
            filter_shape=(192, 192, 1, 1),
            image_shape=(batch_size, 192, 16, 16),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer5 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer4.output,
            filter_shape=(192, 192, 3, 3),
            image_shape=(batch_size, 192, 16, 16),
            ssample=(2,2),
            bordermode='half',
            p=layer_ndo_p
        )
    
        layer6 = myConvLayer(
            rng,
             is_train=training_enabled,
            input_data=layer5.output,
            filter_shape=(192, 192, 3, 3),
            image_shape=(batch_size, 192, 8, 8),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer7 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer6.output,
            filter_shape=(192, 192, 1, 1),
            image_shape=(batch_size, 192, 8, 8),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer8 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer7.output,
            filter_shape=(10, 192, 1, 1),
            image_shape=(batch_size, 192, 8, 8),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )

        global_average=layer8.output.mean(axis=(2,3))
    
        softmax_layer=SoftmaxWrapper(input_data=global_average, n_in=10, n_out=10)
        
        self.errors = softmax_layer.errors(y_data)
    
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
        self.cost = (softmax_layer.negative_log_likelihood(y_data) + L2_reg * L2_sqr)
        
        self.params = layer8.params + layer7.params + layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
        
        self.input = X_data
        self.y = y_data
        
class ModelA_AllCNN(object):
    
    def __init__(self, rng, X_data, y_data, batch_size, training_enabled, layer_ndo_p, L2_reg):

        layer0 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=X_data,
            filter_shape=(96, 3, 5, 5),
            image_shape=(batch_size, 3, 32, 32),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer1 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer0.output,
            filter_shape=(96, 96, 3, 3),
            image_shape=(batch_size, 96, 32, 32),
            ssample=(2,2),
            bordermode='half',
            p=layer_ndo_p
        )
    
        layer2 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer1.output,
            filter_shape=(192, 96, 5, 5),
            image_shape=(batch_size, 96, 16, 16),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer3 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer2.output,
            filter_shape=(192, 192, 3, 3),
            image_shape=(batch_size, 192, 16, 16),
            ssample=(2,2),
            bordermode='half',
            p=layer_ndo_p
        )
    
        layer4 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer3.output,
            filter_shape=(192, 192, 3, 3),
            image_shape=(batch_size, 192, 8, 8),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer5 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer4.output,
            filter_shape=(192, 192, 1, 1),
            image_shape=(batch_size, 192, 8, 8),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer6 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer5.output,
            filter_shape=(10, 192, 1, 1),
            image_shape=(batch_size, 192, 8, 8),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        # make sure this is what global averaging does
        global_average=layer6.output.mean(axis=(2,3))
        
        softmax_layer=SoftmaxWrapper(input_data=global_average, n_in=10, n_out=10)
        
        self.errors = softmax_layer.errors(y_data)
        
        L2_sqr = (
                    (layer0.W ** 2).sum()
                    +(layer1.W**2).sum()
                    +(layer2.W**2).sum()
                    +(layer3.W**2).sum()
                    +(layer4.W**2).sum()
                    +(layer5.W**2).sum()
                    +(layer6.W**2).sum()
        )
    
        # the cost we minimize during training is the NLL of the model
        self.cost = (softmax_layer.negative_log_likelihood(y_data) + L2_reg * L2_sqr)
        
        self.params = layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
        
        self.input = X_data
        self.y = y_data
        

class ModelA_AllCNN_BN(object):
    def __init__(self, rng, X_data, y_data, batch_size, training_enabled, layer_ndo_p, L2_reg):

        layer0 = myConvLayerBN(
            rng,
            is_train=training_enabled,
            input_data=X_data,
            filter_shape=(96, 3, 5, 5),
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
            ssample=(2,2),
            bordermode='half',
            p=layer_ndo_p
        )
    
        layer2 = myConvLayerBN(
            rng,
            is_train=training_enabled,
            input_data=layer1.output,
            filter_shape=(192, 96, 5, 5),
            image_shape=(batch_size, 96, 16, 16),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer3 = myConvLayerBN(
            rng,
            is_train=training_enabled,
            input_data=layer2.output,
            filter_shape=(192, 192, 3, 3),
            image_shape=(batch_size, 192, 16, 16),
            ssample=(2,2),
            bordermode='half',
            p=layer_ndo_p
        )
    
        layer4 = myConvLayerBN(
            rng,
            is_train=training_enabled,
            input_data=layer3.output,
            filter_shape=(192, 192, 3, 3),
            image_shape=(batch_size, 192, 8, 8),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer5 = myConvLayerBN(
            rng,
            is_train=training_enabled,
            input_data=layer4.output,
            filter_shape=(192, 192, 1, 1),
            image_shape=(batch_size, 192, 8, 8),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer6 = myConvLayerBN(
            rng,
            is_train=training_enabled,
            input_data=layer5.output,
            filter_shape=(10, 192, 1, 1),
            image_shape=(batch_size, 192, 8, 8),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        # make sure this is what global averaging does
        global_average=layer6.output.mean(axis=(2,3))
        
        softmax_layer=SoftmaxWrapper(input_data=global_average, n_in=10, n_out=10)
        
        self.errors = softmax_layer.errors(y_data)
        
        L2_sqr = (
                    (layer0.W ** 2).sum()
                    +(layer1.W**2).sum()
                    +(layer2.W**2).sum()
                    +(layer3.W**2).sum()
                    +(layer4.W**2).sum()
                    +(layer5.W**2).sum()
                    +(layer6.W**2).sum()
        )
    
        # the cost we minimize during training is the NLL of the model
        self.cost = (softmax_layer.negative_log_likelihood(y_data) + L2_reg * L2_sqr)
        
        self.params = layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
        
        self.input = X_data
        self.y = y_data

class ModelB_AllCNN_BN(object):
    def __init__(self, rng, X_data, y_data, batch_size, training_enabled, layer_ndo_p, L2_reg):
        
        layer0 = myConvLayerBN(
            rng,
            is_train=training_enabled,
            input_data=X_data,
            filter_shape=(96, 3, 5, 5),
            image_shape=(batch_size, 3, 32, 32),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer1 = myConvLayerBN(
            rng,
            is_train=training_enabled,
            input_data=layer0.output,
            filter_shape=(96, 96, 1, 1),
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
            filter_shape=(192, 96, 5, 5),
            image_shape=(batch_size, 96, 16, 16),
            ssample=(1,1),
            bordermode='half',
            p=1.0
        )
    
        layer4 = myConvLayerBN(
            rng,
            is_train=training_enabled,
            input_data=layer3.output,
            filter_shape=(192, 192, 1, 1),
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

        global_average=layer8.output.mean(axis=(2,3))
    
        softmax_layer=SoftmaxWrapper(input_data=global_average, n_in=10, n_out=10)
        
        self.errors = softmax_layer.errors(y_data)
    
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
        self.cost = (softmax_layer.negative_log_likelihood(y_data) + L2_reg * L2_sqr)
        
        self.params = layer8.params + layer7.params + layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
        
        self.input = X_data
        self.y = y_data


class ModelC_AllCNN_BN(object):
    def __init__(self, rng, X_data, y_data, batch_size, training_enabled, layer_ndo_p, L2_reg):

        layer0 = myConvLayerBN(
            rng,
            is_train=training_enabled,
            input_data=X_data,
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
        
        self.errors = softmax_layer.errors(y_data)
    
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
        self.cost = (softmax_layer.negative_log_likelihood(y_data) + L2_reg*L2_sqr)
        
        self.params = layer8.params + layer7.params + layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

        self.input = X_data
        self.y = y_data


class Large_AllCNN(object):

    def __init__(self, rng, X_data, y_data, batch_size, training_enabled, layer_ndo_p, L2_reg, lrelu_alpha):

        layer0 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=X_data, ## extreme augmented data input
        filter_shape=(320, 3, 2, 2),
        image_shape=(batch_size, 3, 126, 126),
        ssample=(1,1),
        bordermode='valid',
        p=1.0,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer1 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer0.output,
        filter_shape=(320, 320, 2, 2),
        image_shape=(batch_size, 320, 125, 125),
        ssample=(1,1),
        bordermode='valid',
        p=1.0,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer2 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer1.output,
        filter_shape=(320, 320, 2, 2),
        image_shape=(batch_size, 320, 124, 124),
        ssample=(2,2),
        bordermode='valid',
        p=1.0,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer3 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer2.output,
        filter_shape=(640, 320, 2, 2),
        image_shape=(batch_size, 320, 62, 62),
        ssample=(1,1),
        bordermode='valid',
        p=0.9,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer4 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer3.output,
        filter_shape=(640, 640, 2, 2),
        image_shape=(batch_size, 640, 61, 61),
        ssample=(1,1),
        bordermode='valid',
        p=0.9,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer5 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer4.output,
        filter_shape=(640, 640, 2, 2),
        image_shape=(batch_size, 640, 60, 60),
        ssample=(2,2),
        bordermode='valid',
        p=1.0,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer6 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer5.output,
        filter_shape=(960, 640, 2, 2),
        image_shape=(batch_size, 640, 30, 30),
        ssample=(1,1),
        bordermode='valid',
        p=0.8,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer7 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer6.output,
        filter_shape=(960, 960, 2, 2),
        image_shape=(batch_size, 960, 29, 29),
        ssample=(1,1),
        bordermode='valid',
        p=0.8,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer8 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer7.output,
        filter_shape=(960, 960, 2, 2),
        image_shape=(batch_size, 960, 28, 28),
        ssample=(2,2),
        bordermode='valid',
        p=1.0,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer9 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer8.output,
        filter_shape=(1280, 960, 2, 2),
        image_shape=(batch_size, 960, 14, 14),
        ssample=(1,1),
        bordermode='valid',
        p=0.7,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer10 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer9.output,
        filter_shape=(1280, 1280, 2, 2),
        image_shape=(batch_size, 1280, 13, 13),
        ssample=(1,1),
        bordermode='valid',
        p=0.7,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer11 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer10.output,
        filter_shape=(1280, 1280, 2, 2),
        image_shape=(batch_size, 1280, 12, 12),
        ssample=(2,2),
        bordermode='valid',
        p=1.0,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer12 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer11.output,
        filter_shape=(1600, 1280, 2, 2),
        image_shape=(batch_size, 1280, 6, 6),
        ssample=(1,1),
        bordermode='valid',
        p=0.6,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer13 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer12.output,
        filter_shape=(1600, 1600, 2, 2),
        image_shape=(batch_size, 1600, 5, 5),
        ssample=(1,1),
        bordermode='valid',
        p=0.6,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer14 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer13.output,
        filter_shape=(1600, 1600, 2, 2),
        image_shape=(batch_size, 1600, 4, 4),
        ssample=(2,2),
        bordermode='valid',
        p=1.0,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer15 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer14.output,
        filter_shape=(1920, 1600, 2, 2),
        image_shape=(batch_size, 1600, 2, 2),
        ssample=(1,1),
        bordermode='valid',
        p=0.5,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer16 = myConvLayer(
        rng,
        is_train=training_enabled,
        input_data=layer15.output,
        filter_shape=(1920, 1920, 1, 1),
        image_shape=(batch_size, 1920, 1, 1),
        ssample=(1,1),
        bordermode='valid',
        p=0.5,
        alpha=0.5  ##leaky relu
        )
    
        layer17 = myConvLayer(
            rng,
            is_train=training_enabled,
            input_data=layer16.output,
            filter_shape=(10,1920,1,1),
            image_shape=(batch_size,1920,1,1),
            ssample=(1,1),
            bordermode='valid',
            p=1.0,
            alpha=lrelu_alpha ##leaky relu
        )
        
        softmax_input = layer17.output.flatten(2)
        
        softmax_layer=SoftmaxWrapper(input_data=softmax_input, n_in=10, n_out=10)
        
        self.errors = softmax_layer.errors(y_data)
        
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
                    +(layer9.W**2).sum()
                    +(layer10.W**2).sum()
                    +(layer11.W**2).sum()
                    +(layer12.W**2).sum()
                    +(layer13.W**2).sum()
                    +(layer14.W**2).sum()
                    +(layer15.W**2).sum()
                    +(layer16.W**2).sum()
                    +(layer17.W**2).sum()
    
        )
    
        # the cost we minimize during training is the NLL of the model
        self.cost = (softmax_layer.negative_log_likelihood(y_data) + L2_reg*L2_sqr)

        self.params = layer17.params + layer16.params + layer15.params + layer14.params + layer13.params + layer12.params + layer11.params + layer10.params + layer9.params + layer8.params + layer7.params + layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
        
        self.input = X_data
        self.y = y_data

class Large_AllCNN_BN(object):

    def __init__(self, rng, X_data, y_data, batch_size, training_enabled, layer_ndo_p, L2_reg, lrelu_alpha):

        layer0 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=X_data, ## extreme augmented data input
        filter_shape=(320, 3, 2, 2),
        image_shape=(batch_size, 3, 126, 126),
        ssample=(1,1),
        bordermode='valid',
        p=1.0,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer1 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer0.output,
        filter_shape=(320, 320, 2, 2),
        image_shape=(batch_size, 320, 125, 125),
        ssample=(1,1),
        bordermode='valid',
        p=1.0,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer2 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer1.output,
        filter_shape=(320, 320, 2, 2),
        image_shape=(batch_size, 320, 124, 124),
        ssample=(2,2),
        bordermode='valid',
        p=1.0,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer3 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer2.output,
        filter_shape=(640, 320, 2, 2),
        image_shape=(batch_size, 320, 62, 62),
        ssample=(1,1),
        bordermode='valid',
        p=0.9,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer4 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer3.output,
        filter_shape=(640, 640, 2, 2),
        image_shape=(batch_size, 640, 61, 61),
        ssample=(1,1),
        bordermode='valid',
        p=0.9,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer5 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer4.output,
        filter_shape=(640, 640, 2, 2),
        image_shape=(batch_size, 640, 60, 60),
        ssample=(2,2),
        bordermode='valid',
        p=1.0,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer6 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer5.output,
        filter_shape=(960, 640, 2, 2),
        image_shape=(batch_size, 640, 30, 30),
        ssample=(1,1),
        bordermode='valid',
        p=0.8,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer7 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer6.output,
        filter_shape=(960, 960, 2, 2),
        image_shape=(batch_size, 960, 29, 29),
        ssample=(1,1),
        bordermode='valid',
        p=0.8,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer8 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer7.output,
        filter_shape=(960, 960, 2, 2),
        image_shape=(batch_size, 960, 28, 28),
        ssample=(2,2),
        bordermode='valid',
        p=1.0,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer9 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer8.output,
        filter_shape=(1280, 960, 2, 2),
        image_shape=(batch_size, 960, 14, 14),
        ssample=(1,1),
        bordermode='valid',
        p=0.7,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer10 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer9.output,
        filter_shape=(1280, 1280, 2, 2),
        image_shape=(batch_size, 1280, 13, 13),
        ssample=(1,1),
        bordermode='valid',
        p=0.7,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer11 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer10.output,
        filter_shape=(1280, 1280, 2, 2),
        image_shape=(batch_size, 1280, 12, 12),
        ssample=(2,2),
        bordermode='valid',
        p=1.0,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer12 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer11.output,
        filter_shape=(1600, 1280, 2, 2),
        image_shape=(batch_size, 1280, 6, 6),
        ssample=(1,1),
        bordermode='valid',
        p=0.6,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer13 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer12.output,
        filter_shape=(1600, 1600, 2, 2),
        image_shape=(batch_size, 1600, 5, 5),
        ssample=(1,1),
        bordermode='valid',
        p=0.6,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer14 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer13.output,
        filter_shape=(1600, 1600, 2, 2),
        image_shape=(batch_size, 1600, 4, 4),
        ssample=(2,2),
        bordermode='valid',
        p=1.0,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer15 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer14.output,
        filter_shape=(1920, 1600, 2, 2),
        image_shape=(batch_size, 1600, 2, 2),
        ssample=(1,1),
        bordermode='valid',
        p=0.5,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer16 = myConvLayerBN(
        rng,
        is_train=training_enabled,
        input_data=layer15.output,
        filter_shape=(1920, 1920, 1, 1),
        image_shape=(batch_size, 1920, 1, 1),
        ssample=(1,1),
        bordermode='valid',
        p=0.5,
        alpha=lrelu_alpha  ##leaky relu
        )
    
        layer17 = myConvLayerBN(
            rng,
            is_train=training_enabled,
            input_data=layer16.output,
            filter_shape=(10,1920,1,1),
            image_shape=(batch_size,1920,1,1),
            ssample=(1,1),
            bordermode='valid',
            p=1.0,
            alpha=lrelu_alpha ##leaky relu
        )
        
        # no global averaging required in this case
        softmax_input = layer17.output.flatten(2)
        
        softmax_layer=SoftmaxWrapper(input_data=softmax_input, n_in=10, n_out=10)
        
        self.errors = softmax_layer.errors(y_data)
        
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
                    +(layer9.W**2).sum()
                    +(layer10.W**2).sum()
                    +(layer11.W**2).sum()
                    +(layer12.W**2).sum()
                    +(layer13.W**2).sum()
                    +(layer14.W**2).sum()
                    +(layer15.W**2).sum()
                    +(layer16.W**2).sum()
                    +(layer17.W**2).sum()
    
        )
    
        # the cost we minimize during training is the NLL of the model
        self.cost = (softmax_layer.negative_log_likelihood(y_data) + L2_reg*L2_sqr)

        self.params = layer17.params + layer16.params + layer15.params + layer14.params + layer13.params + layer12.params + layer11.params + layer10.params + layer9.params + layer8.params + layer7.params + layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
        
        self.input = X_data
        self.y = y_data
