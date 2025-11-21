import keras
from keras.layers import Conv2D, Add, Multiply, BatchNormalization, Activation
from keras import regularizers
from global_var import myModelConfig


def score_refine_module(input_feature_map, map_name=None):

    if keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(256, (1, 1), kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(input_feature_map)
    x = BatchNormalization(axis=bn_axis, name=map_name+'bn_SR_map')(x)
    x = Activation('relu')(x)
    # score head:
    score_map_s1 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(x)
    score_map_s1 = BatchNormalization(axis=bn_axis, name=map_name+'bn_s1_1')(score_map_s1)
    score_map_s1 = Activation('relu')(score_map_s1)

    score_map_s2 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(x)
    score_map_s2 = BatchNormalization(axis=bn_axis, name=map_name+'bn_s2_1')(score_map_s2)
    score_map_s2 = Activation('relu')(score_map_s2)

    score_map_s1s2 = Add()([score_map_s1, score_map_s2])

    score_map_s1 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(score_map_s1s2)
    score_map_s1 = BatchNormalization(axis=bn_axis, name=map_name+'bn_s1_2')(score_map_s1)
    score_map_s1 = Activation('relu')(score_map_s1)

    score_map_s2 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(score_map_s1s2)
    score_map_s2 = BatchNormalization(axis=bn_axis, name=map_name+'bn_s2_2')(score_map_s2)
    score_map_s2 = Activation('relu')(score_map_s2)

    score_map_s1s2 = Add()([score_map_s1, score_map_s2])

    score_map_s1 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(score_map_s1s2)
    score_map_s1 = BatchNormalization(axis=bn_axis, name=map_name+'bn_s1_3')(score_map_s1)
    score_map_s1 = Activation('relu')(score_map_s1)

    score_map_s2 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(score_map_s1s2)
    score_map_s2 = BatchNormalization(axis=bn_axis, name=map_name+'bn_s2_3')(score_map_s2)
    score_map_s2 = Activation('relu')(score_map_s2)

    score_map_s1s2 = Add()([score_map_s1, score_map_s2])

    score_map_s1 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(score_map_s1s2)
    score_map_s1 = BatchNormalization(axis=bn_axis, name=map_name+'bn_s1_4')(score_map_s1)
    score_map_s1 = Activation('relu')(score_map_s1)

    score_map_s2 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(score_map_s1s2)
    score_map_s2 = BatchNormalization(axis=bn_axis, name=map_name+'bn_s2_4')(score_map_s2)
    score_map_s2 = Activation('relu')(score_map_s2)

    score_map_s1s2 = Add()([score_map_s1, score_map_s2])

    score_map = Conv2D(5, (1, 1), padding='same', activation="relu", kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(score_map_s1s2)

    # locate head
    locate_head = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(x)
    locate_head = BatchNormalization(axis=bn_axis, name=map_name+'bn_l1_1')(locate_head)
    locate_head = Activation('relu')(locate_head)

    locate_head = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(locate_head)
    locate_head = BatchNormalization(axis=bn_axis, name=map_name+'bn_l1_2')(locate_head)
    locate_head = Activation('relu')(locate_head)

    locate_head = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(locate_head)
    locate_head = BatchNormalization(axis=bn_axis, name=map_name+'bn_l1_3')(locate_head)
    locate_head = Activation('relu')(locate_head)

    locate_map = Conv2D(1, (1, 1), padding='same', activation="sigmoid", kernel_initializer='he_normal',
                        name=map_name + "_locate",
                        kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(locate_head)

    refined_map = Multiply(name=map_name+"refine_score_map")([score_map, locate_map])

    return refined_map, locate_map
