from keras.models import Model
from keras.layers import Conv1D, BatchNormalization, Activation
from keras.layers import Input, concatenate, MaxPooling1D, UpSampling1D
from keras.regularizers import l2


class Models:

    @staticmethod
    def model_unet(input_shape,num_classes=2,base_filters=16,wd=0):
        """
        base_filters=16 (reduce size of model)
        """

        inputs = Input(shape=input_shape)
        #model input stage
        batch_norm0 = BatchNormalization(fused=False)(inputs)
        conv0 = Conv1D(filters=base_filters, kernel_size=(49), padding='same', use_bias=False, strides=1,
                       kernel_initializer='he_uniform',kernel_regularizer=l2(wd))(batch_norm0)
        #model main stage
        Module1 = Models._UModule(inputs=conv0,num_filters=base_filters,wd=wd)
        Pool1 = MaxPooling1D(pool_size=(2))(Module1)
        Module2 = Models._UModule(inputs=Pool1,num_filters=base_filters*2,wd=wd)
        Pool2 = MaxPooling1D(pool_size=(2))(Module2)
        Module3 = Models._UModule(inputs=Pool2,num_filters=base_filters*4,wd=wd)
        Pool3 = MaxPooling1D(pool_size=(2))(Module3)
        Module4 = Models._UModule(inputs=Pool3,num_filters=base_filters*8,wd=wd)
        Pool4 = MaxPooling1D(pool_size=(2))(Module4)
        
        Module5 = Models._UModule(inputs=Pool4,num_filters=base_filters*16,wd=wd)
        
        up6 = concatenate([UpSampling1D(size=(2))(Module5), Module4])
        Module6 = Models._UModule(inputs=up6,num_filters=base_filters*8,wd=wd)
        up7 = concatenate([UpSampling1D(size=(2))(Module6), Module3])
        Module7 = Models._UModule(inputs=up7,num_filters=base_filters*4,wd=wd)
        up8 = concatenate([UpSampling1D(size=(2))(Module7), Module2])
        Module8 = Models._UModule(inputs=up8,num_filters=base_filters*2,wd=wd)
        up9 = concatenate([UpSampling1D(size=(2))(Module8), Module1])
        Module9 = Models._UModule(inputs=up9,num_filters=base_filters,wd=wd)
    
        #model output stage
        ModuleOut = BatchNormalization(center=True, scale=True,fused=False)(Module9)
        ModuleOut = Activation('relu')(ModuleOut)
        ModuleOut = Conv1D(filters=num_classes, kernel_size=(1), padding='same', use_bias=False, strides=1,
                           kernel_initializer='he_uniform',kernel_regularizer=l2(wd))(ModuleOut)
        
        ModuleOut = Activation('softmax')(ModuleOut)
        model = Model(inputs=inputs, outputs=ModuleOut)
        return model
    
    @staticmethod
    def _UModule(inputs, num_filters, wd):
        x = BatchNormalization(center=True, scale=True, fused=False)(inputs)
        x = Activation('relu')(x)
        x = Conv1D(filters=num_filters,
                   kernel_size=(9), 
                   padding='same', 
                   kernel_initializer='he_uniform', 
                   strides=1,
                   use_bias=False,
                   kernel_regularizer=l2(wd))(x)
        x = BatchNormalization(center=True, scale=True,fused=False)(x)
        x = Activation('relu')(x)
        x = Conv1D(filters=num_filters, 
                   kernel_size=(9), 
                   padding='same', 
                   kernel_initializer='he_uniform', 
                   strides=1,
                   use_bias=False,
                   kernel_regularizer=l2(wd))(x)
        return x
