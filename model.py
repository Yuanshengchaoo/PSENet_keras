import tensorflow as tf
import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Conv2D, Add, GlobalAveragePooling2D, Concatenate, MaxPooling2D
from keras import regularizers
from SR_Module import score_refine_module
from keras.layers import Layer, Lambda, BatchNormalization, Activation


def combine_siamese_results(output_score_map_a, output_score_map_b):
    # 该函数输出5*3个值，A网络的5个分数，B网络的5个分数以及AB两网络的5个分数差距

    # score A:
    A_vector = Concatenate()([GlobalAveragePooling2D()(output_score_map_a[i]) for i in range(5)])
    A_score = Dense(5, activation=None, name="scoreA", kernel_initializer='he_normal')(A_vector)

    # score B:
    B_vector = Concatenate()([GlobalAveragePooling2D()(output_score_map_b[i]) for i in range(5)])
    B_score = Dense(5, activation=None, name="scoreB", kernel_initializer='he_normal')(B_vector)

    # score gap:
    siamese_map = [Add()([output_score_map_a[i], output_score_map_b[i]]) for i in range(5)]
    siamese_vector = Concatenate()([GlobalAveragePooling2D()(siamese_map[i]) for i in range(5)])
    siamese_score = Dense(5, activation=None, name="scoreSiam", kernel_initializer='he_normal')(siamese_vector)

    return A_score, B_score, siamese_score


def upsampleing(pair):
    deep = pair[0]
    shallow = pair[1]
    shallow_shape = tf.shape(shallow)
    deep_up = tf.image.resize(deep, [shallow_shape[1], shallow_shape[2]], method="nearest")
    return deep_up


def PSENet(myModelConfig):
    # define single network
    image_input = Input(shape=(myModelConfig.img_height, myModelConfig.img_width, 3), name="single_input")

    if keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    base_model = ResNet50(weights="imagenet", input_shape=(800, 1024, 3), input_tensor=image_input,
                          include_top=False)

    for layer in base_model.layers:
        if isinstance(layer, keras.layers.DepthwiseConv2D):
            layer.add_loss(keras.regularizers.l2(myModelConfig.weight_decay)(layer.depthwise_kernel))
        elif isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense)):
            layer.add_loss(keras.regularizers.l2(myModelConfig.weight_decay)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(keras.regularizers.l2(myModelConfig.weight_decay)(layer.bias))

    # FPN feature taps align with Keras 3 ResNet50 layer names
    p3 = base_model.get_layer("conv3_block4_out").output
    p4 = base_model.get_layer("conv4_block6_out").output
    p5 = base_model.get_layer("conv5_block3_out").output

    p6 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(base_model.output)
    p6 = BatchNormalization(axis=bn_axis, name='bn_p6_conv1')(p6)
    p6 = Activation('relu')(p6)

    p6 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p6)
    p6 = BatchNormalization(axis=bn_axis, name='bn_p6_conv2')(p6)
    p6 = Activation('relu')(p6)
    p6 = MaxPooling2D(pool_size=(2, 2), padding='same')(p6)

    p7 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p6)
    p7 = BatchNormalization(axis=bn_axis, name='bn_p7_conv1')(p7)
    p7 = Activation('relu')(p7)
    p7 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p7)
    p7 = BatchNormalization(axis=bn_axis, name='bn_p7_conv2')(p7)
    p7 = Activation('relu')(p7)
    p7 = MaxPooling2D(pool_size=(2, 2), padding='same')(p7)

    # P7+P6
    p7_up = Lambda(upsampleing)([p7, p6])
    p6 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p6)
    p6 = BatchNormalization(axis=bn_axis, name='bn_p6_reduce')(p6)
    p6 = Activation('relu')(p6)

    p6_map = Add()([p6, p7_up])
    p6 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p6_map)
    p6 = BatchNormalization(axis=bn_axis, name='bn_p6_out')(p6)
    p6 = Activation('relu')(p6)

    # P6+P5
    p6_up = Lambda(upsampleing)([p6, p5])
    p5 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p5)
    p5 = BatchNormalization(axis=bn_axis, name='bn_p5_reduce')(p5)
    p5 = Activation('relu')(p5)

    p5_map = Add()([p5, p6_up])
    p5 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p5_map)
    p5 = BatchNormalization(axis=bn_axis, name='bn_p5_out')(p5)
    p5 = Activation('relu')(p5)

    # P5+P4
    p5_up = Lambda(upsampleing)([p5, p4])
    p4 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p4)
    p4 = BatchNormalization(axis=bn_axis, name='bn_p4_reduce')(p4)
    p4 = Activation('relu')(p4)

    p4_map = Add()([p4, p5_up])
    p4 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p4_map)
    p4 = BatchNormalization(axis=bn_axis, name='bn_p4_out')(p4)
    p4 = Activation('relu')(p4)

    # P4+P3
    p4_up = Lambda(upsampleing)([p4, p3])
    p3 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p3)
    p3 = BatchNormalization(axis=bn_axis, name='bn_p3_reduce')(p3)
    p3 = Activation('relu')(p3)

    p3_map = Add()([p3, p4_up])
    p3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(myModelConfig.weight_decay))(p3_map)
    p3 = BatchNormalization(axis=bn_axis, name='bn_p3_out')(p3)
    p3 = Activation('relu')(p3)

    p3_score_map, p3_locate_map = score_refine_module(p3, "p3")
    p4_score_map, p4_locate_map = score_refine_module(p4, "p4")
    p5_score_map, p5_locate_map = score_refine_module(p5, "p5")
    p6_score_map, p6_locate_map = score_refine_module(p6, "p6")
    p7_score_map, p7_locate_map = score_refine_module(p7, "p7")

    single_model = Model(inputs=image_input,
                         outputs=[p3_score_map, p4_score_map, p5_score_map, p6_score_map, p7_score_map,
                                  p3_locate_map, p4_locate_map, p5_locate_map, p6_locate_map, p7_locate_map],
                         name="single_model")

    # define siamese network
    input_a = Input(shape=(myModelConfig.img_height, myModelConfig.img_width, 3), name="input_a")
    input_b = Input(shape=(myModelConfig.img_height, myModelConfig.img_width, 3), name="input_b")

    output_a = single_model(input_a)
    output_b = single_model(input_b)

    A_score, B_score, siamese_score = combine_siamese_results(output_a[:5], output_b[:5])

    # siamese模型的总输出包括：
    # A_score 5个分, B_score 5个分, siamese_score 5个分, A_locate_map x 5, B_locate_map x 5
    # 5个分的顺序是area, ery, sca, ind, pasi

    output_list = [A_score, B_score, siamese_score]

    output_list.extend(output_a[5:])
    output_list.extend(output_b[5:])

    siamese_model = Model(inputs=[input_a, input_b], outputs=output_list, name="siamese")

    return siamese_model
