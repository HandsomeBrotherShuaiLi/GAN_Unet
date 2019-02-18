import numpy as np
from functools import partial
from keras import backend as K
from keras.engine import Input,Model
from keras.layers import Conv3D,MaxPooling3D,UpSampling3D,Activation
from keras.layers import BatchNormalization,PReLU,Deconvolution3D
from keras.layers.merge import concatenate
from keras.optimizers import Adam,SGD,RMSprop
from utils.metrics import get_label_dice_coefficient_function,dice_coefficient_loss,dice_coefficient

K.set_image_data_format("channels_first")
"""
TODO: figure out the best loss function and metrics
"""
class Original_Unet_3D(object):
    def __init__(self,
                 image_data_format,
                 input_shape,
                 metrics=dice_coefficient,
                 pool_size=(2,2,2),
                 n_labels=1,
                 initial_lr=1e-5,
                 deconvolution=False,
                 depth=4,
                 n_base_filters=32,
                 include_label_wise_dice_coef=False,
                 batch_normalization=False,
                 activation_name='sigmoid',
                 optimizer='Adam'
                 ):
        """
        init the model
        :param image_data_format:image_data_format,eg:'channels_first'如果 data_format='channels_first'， 输入 5D 张量，
        尺寸为 (samples, channels, conv_dim1, conv_dim2, conv_dim3)。
        如果 data_format='channels_last'， 输入 5D 张量，尺寸为 (samples, conv_dim1, conv_dim2, conv_dim3, channels)。
        :param input_shape: Shape of the input data (samples,n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
                divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth
        :param metrics:loss function list
        :param pool_size:pool size for maxpooling operation
        :param n_labels:Number of binary labels that the model is learning.
        :param initial_lr:initial learning rate
        :param deconvolution: If set to True, the model will use transpose convolution(deconvolution) instead of up-sampling. This
                increases the amount memory required during training.Default parameter is False
        :param depth:indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
               layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
        :param n_base_filters:The number of filters that the first layer in the convolution network will have. Following
                layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
                to train the model.
        :param include_label_wise_dice_coef:If True and n_labels is greater than 1, model will report the dice
                coefficient for each label as metric.
        :param batch_normalization:use bn or not
        :param activation_name:activation function name

        """
        self.image_data_format=image_data_format
        self.input_shape=input_shape
        self.metrics=metrics
        self.pool_size=pool_size
        self.nlabels=n_labels
        self.initial_lr=initial_lr
        self.deconvolution=deconvolution
        self.depth=depth
        self.n_base_filters=n_base_filters
        self.include_label_wise_dice_coef=include_label_wise_dice_coef
        self.batch_normalization=batch_normalization
        self.activation_name=activation_name
        self.optimizer=optimizer

    def create_convolution_block(self,input_layer,n_filters,batch_normalization=False,
                                 kernel=(3,3,3),activation='relu',padding='same',strides=(1,1,1),
                                 instance_normlization=False):
        """

        :param input_layer:
        :param n_filters:
        :param batch_normalization:
        :param kernel:
        :param activation:
        :param padding:
        :param strides:
        :param instance_normlization:
        :return:
        """
        # print('create block shape input shape',input_layer.shape)
        layer=Conv3D(n_filters,kernel,padding=padding,strides=strides)(input_layer)
        # print('after conv shape',layer.shape)
        if batch_normalization:
            layer=BatchNormalization(axis=1)(layer)
        elif instance_normlization:
            from keras_contrib.layers.normalization import InstanceNormalization
            layer=InstanceNormalization(axis=1)(layer)
        layer=Activation(activation=activation)(layer)
        return layer
    def get_up_convolution(self,n_filtes,pool_size,kernel_size=(2,2,2),strides=(2,2,2),deconvolution=False):
        """
        featuren map [w,h] ----> [nw,wh]
        but deconvolution repeats 0,upsampling repeats the original data
        :param n_filtes:
        :param pool_size: int, or tuple of 2 integers. The upsampling factors for rows and columns.
        :param kernel_size:An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
        Can be a single integer to specify the same value for all spatial dimensions.
        :param strides:
        :param deconvolution:
        :return:
        """
        if deconvolution:
            return Deconvolution3D(filters=n_filtes,kernel_size=kernel_size,strides=strides)
        else:
            return UpSampling3D(size=pool_size)

    def compute_level_output_shape(self,n_filters,depth,pool_size,img_shape):
        """
         Each level has a particular output shape based on the number of filters used in that level and the depth or number
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node
        """
        return tuple([None,n_filters]+np.asarray(np.divide(img_shape,np.power(pool_size,depth)),dtype=np.int32).tolist())
    def model(self):
        """
        build unet 3d
        :return: unet3d model
        """

        input_layer=Input(self.input_shape)
        layer=input_layer
        # print(layer.shape)
        level=[]

        for layer_depth in range(self.depth):
            layer1=self.create_convolution_block(
                input_layer=layer,
                n_filters=self.n_base_filters*(2**layer_depth),
                batch_normalization=self.batch_normalization,
                padding='same'
            )
            # print('layer1 shape:',layer1.shape)
            layer2=self.create_convolution_block(
                input_layer=layer1,
                n_filters=self.n_base_filters*(2**(layer_depth))*2,
                batch_normalization=self.batch_normalization,
                padding='same'
            )
            # print('layer2 shape',layer2.shape)
            if layer_depth<self.depth-1:
                layer=MaxPooling3D(pool_size=self.pool_size)(layer2)
                level.append([layer1,layer2,layer])
            else:
                layer=layer2
                level.append([layer1,layer2])
            # print(layer.shape)

        for layer_depth in range(self.depth-2,-1,-1):
            up_convolution=self.get_up_convolution(
                pool_size=self.pool_size,
                deconvolution=self.deconvolution,
                n_filtes=layer._keras_shape[1]
            )(layer)
            # print('up_convolution shape',up_convolution.shape)
            # print('level layerdepth',level[layer_depth][1].shape)
            concat=concatenate([up_convolution,level[layer_depth][1]],axis=1)
            layer=self.create_convolution_block(
                n_filters=level[layer_depth][1]._keras_shape[1],
                input_layer=concat,
                batch_normalization=self.batch_normalization
            )
            layer=self.create_convolution_block(
                input_layer=layer,
                n_filters=level[layer_depth][1]._keras_shape[1],
                batch_normalization=self.batch_normalization
            )

        final_convolution=Conv3D(self.nlabels,(1,1,1))(layer)
        final=Activation(activation=self.activation_name)(final_convolution)
        model=Model(inputs=input_layer,outputs=final)

        if not isinstance(self.metrics,list):
            self.metrics=[self.metrics]
        if self.include_label_wise_dice_coef and self.nlabels>1:
            label_wise_dice_metrics=[
                get_label_dice_coefficient_function(i) for i in range(self.nlabels)
            ]
            if self.metrics:
                self.metrics=self.metrics+label_wise_dice_metrics
            else:
                self.metrics=label_wise_dice_metrics
        else:
            label_wise_dice_metrics='binary_crossentropy'

        res={
            'Adam':Adam,
            'SGD':SGD,
            'RMSprop':RMSprop
        }
        #我使用的是dice_coefficient,2*|pred∩true|/(|pred|+|true|+smooth)
        #smooth平滑系数
        model.compile(
            optimizer=res[self.optimizer](lr=self.initial_lr),
            loss=label_wise_dice_metrics,
            metrics=self.metrics
        )
        return model

if __name__=='__main__':
    m = Original_Unet_3D(
        input_shape=(256,256,256,45),
        image_data_format='channels_first',
        n_labels=1,
        depth=4,
        n_base_filters=32
    ).model()
    print(m.summary())

