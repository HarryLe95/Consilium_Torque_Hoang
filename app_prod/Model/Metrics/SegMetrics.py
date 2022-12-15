from keras import backend as K


# loss and metric suitable to the "segmentation" strategy for the targets/prediction
def loss_Dice_1D(y_true, y_pred):
    TargetClass=1
    tp = K.sum(y_true[:,:,TargetClass] * y_pred[:,:,TargetClass], axis=[0,1]) 
    fp = K.sum(y_pred[:,:,TargetClass], axis=[0,1]) - tp
    fn = K.sum(y_true[:,:,TargetClass], axis=[0,1]) - tp
    return 1.0 - (2*tp + K.epsilon()) / ( 2*tp +  fn + fp + K.epsilon())

def loss_F_beta_1D(Beta):   
    # reference: https://arxiv.org/pdf/1803.11078.pdf
    def loss(y_true, y_pred):
        tp = K.sum(y_true * y_pred, axis=[0,1]) 
        fp = K.sum(y_pred, axis=[0,1]) - tp
        fn = K.sum(y_true, axis=[0,1]) - tp
        W1 = 1.0+Beta**2
        W2 = Beta**2
        return (1.0 - (W1*tp + K.epsilon()) / ( W1*tp +  W2*fn + fp + K.epsilon()))
    return loss

def metric_dice_int_1D(y_true, y_pred):
    smooth = K.epsilon()#1e-12
    TargetClass = 1
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    predicted_mask = K.cast(K.equal(class_id_preds, TargetClass), 'float32')
    actual_mask = K.cast(K.equal(class_id_true, TargetClass), 'float32')
    #get the intersection and sum
    total_class_intersection = K.sum(predicted_mask*actual_mask, axis=[0,1]) #sums over patches and space, this class
    total_class_sum = K.sum(actual_mask + predicted_mask, axis=[0,1]) #sums over patches and space, this class
    #calculate Dice Coefficient for this class
    DC = (2*total_class_intersection + smooth) / (total_class_sum  + smooth)
    return DC

def metric_precision_1D(y_true, y_pred):
    TargetClass = 1
    smooth = K.epsilon()#1e-12
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    predicted_mask = K.cast(K.equal(class_id_preds, TargetClass), 'float32')
    actual_mask = K.cast(K.equal(class_id_true, TargetClass), 'float32')
    total_class_intersection = K.sum(predicted_mask*actual_mask, axis=[0,1])
    total_class_sum = K.sum(predicted_mask, axis=[0,1])
    Precision = (total_class_intersection + smooth) / (total_class_sum  + smooth)
    return Precision

def metric_recall_1D(y_true, y_pred):
    TargetClass = 1
    smooth = K.epsilon()#1e-12
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    predicted_mask = K.cast(K.equal(class_id_preds, TargetClass), 'float32')
    actual_mask = K.cast(K.equal(class_id_true, TargetClass), 'float32')
    total_class_intersection = K.sum(predicted_mask*actual_mask, axis=[0,1])
    total_class_sum = K.sum(actual_mask, axis=[0,1])
    Recall = (total_class_intersection + smooth) / (total_class_sum  + smooth)
    return Recall

