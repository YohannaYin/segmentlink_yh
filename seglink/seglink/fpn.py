# 在segment link上添加fpn网络，对小文本的特征提取更加好
import tensorflow as tf
import tensorflow.contrib.slim as slim
def build_feature_pyramid(self):
    '''
    reference: https://github.com/CharlesShang/FastMaskRCNN
    build P2, P3, P4, P5
    :return: multi-scale feature map
    '''

    feature_pyramid = {}
    with tf.variable_scope('build_feature_pyramid'):
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.rpn_weight_decay)):
            # 首先使用
            feature_pyramid['P5'] = slim.conv2d(self.feature_maps_dict['C5'],
                                                num_outputs=256,
                                                kernel_size=[1, 1],
                                                stride=1,
                                                scope='build_P5')

            feature_pyramid['P6'] = slim.max_pool2d(feature_pyramid['P5'],
                                                    kernel_size=[2, 2], stride=2, scope='build_P6')

            # 下采样，从conv11到conv7层
            # P6 is down sample of P5
            for layer in range(6, 1, -1):
                p, c = feature_pyramid['P' + str(layer + 1)], self.feature_maps_dict['C' + str(layer)]
                up_sample_shape = tf.shape(c)
                # 最近邻插值
                up_sample = tf.image.resize_nearest_neighbor(p, [up_sample_shape[1], up_sample_shape[2]],
                                                             name='build_P%d/up_sample_nearest_neighbor' % layer)
                # 降维，kernel size = 1*1
                c = slim.conv2d(c, num_outputs=256, kernel_size=[1, 1], stride=1,
                                scope='build_P%d/reduce_dimension' % layer)
                # 直接将上采样的值和c的值做concate
                p = up_sample + c
                # 3×3卷积
                p = slim.conv2d(p, 256, kernel_size=[3, 3], stride=1,
                                padding='SAME', scope='build_P%d/avoid_aliasing' % layer)

                feature_pyramid['P' + str(layer)] = p

    return feature_pyramid