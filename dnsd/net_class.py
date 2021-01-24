from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from PIL import Image
from tensorflow.contrib import layers
from tensorflow.contrib import  slim
import time
import numpy as np
import os


def tanh_out(x):
    return tf.nn.tanh(x) + 1

def vgg_16(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=1,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):







    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1',trainable=False)
    st1=net = slim.max_pool2d(net, [2, 2], scope='pool1')#1/2
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2',trainable=False)  #1/2
    st2 =net = slim.max_pool2d(net, [2, 2], scope='pool2')#1/4
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3',trainable=False)#1/4
    st3 =net = slim.max_pool2d(net, [2, 2], scope='pool3')#1/8
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4',trainable=False)  #1/8
    st4 =net = slim.max_pool2d(net, [2, 2], scope='pool4')#1/16
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5',trainable=False)#1/16
    st5 =net = slim.max_pool2d(net, [2, 2], scope='pool5')#1/32

    # Use conv2d instead of fully_connected layers.
    #net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6',trainable=False)#[?,1,1,4096]
    #net = slim.dropout(net, dropout_keep_prob, is_training=is_training,scope='dropout6')
    #net = slim.conv2d(net, 4096, [1, 1], scope='fc7',trainable=False)#4096  # style, feature



    return st5,st4, st3, st2,st1





def res_block(inp,inp_depth,calc_depth,name,size=3,stride=1):
    #net=slim.conv2d(inp,calc_depth,[1,1],1)
    #shape=tf.shape(inp)
    #inp_depth=shape[-1]
    net = slim.conv2d(inp, calc_depth, [size, size], stride,normalizer_fn=slim.batch_norm,activation_fn=tf.nn.leaky_relu)
    net = slim.conv2d(net, inp_depth, [1, 1], 1,activation_fn=None,normalizer_fn=slim.batch_norm)
    net=tf.add(net,inp,name=name)
    #net=tf.concat([net,inp],axis=-1)
    #net=tf.nn.leaky_relu(net,name=name)

    return net
#***************************************************************************************

def describe_graph(graph_def, show_nodes=False):
    print('Input Feature Nodes: {}'.format(
        [node.name for node in graph_def.node if node.op == 'Placeholder']))
    print('')
    print('Unused Nodes: {}'.format(
        [node.name for node in graph_def.node if 'unused' in node.name]))
    print('')
    print('Output Nodes: {}'.format(
        [node.name for node in graph_def.node if (
                'predictions' in node.name or 'softmax' in node.name)]))
    print('')
    print('Quantization Nodes: {}'.format(
        [node.name for node in graph_def.node if 'quant' in node.name]))
    print('')
    print('Constant Count: {}'.format(
        len([node for node in graph_def.node if node.op == 'Const'])))
    print('')
    print('Variable Count: {}'.format(
        len([node for node in graph_def.node if 'Variable' in node.op])))
    print('')
    print('Identity Count: {}'.format(
        len([node for node in graph_def.node if node.op == 'Identity'])))
    print('', 'Total nodes: {}'.format(len(graph_def.node)), '')

    if show_nodes == True:
        for node in graph_def.node:
            print('Op:{} - Name: {}'.format(node.op, node.name))

class vgg_est_net:

    def create(self):
        with tf.variable_scope('trans_net'):
            with tf.name_scope('input'):
                x = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='input')
                # x_f,y=next_element
                y = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name='label')
                tf.summary.image('input', x, 1)
                tf.summary.image('y', y, 1)

                inp = x
            with tf.name_scope('DS_1'):
                net1 = slim.conv2d(inp, 64, [2,2], 2, padding='same')# ,  normalizer_fn=layers.batch_norm)  # 1/2
                #net1=slim.max_pool2d(net1,[2,2])
            with tf.name_scope('DS_2'):
                net2 = slim.conv2d(net1, 128, [2, 2], 2, padding='same')# ,  normalizer_fn=layers.batch_norm)  # 1/4
                #net2 = slim.max_pool2d(net2, [2, 2])
            with tf.name_scope('DS_3'):
                net3 = slim.conv2d(net2, 256, [2, 2], 2,
                                     padding='same')#   ,normalizer_fn=layers.batch_norm)  # 1/16
                #net3 = slim.max_pool2d(net3, [2, 2])
            with tf.name_scope('DS_4'):
                net4 = slim.conv2d(net3, 512, [2,2], 2,
                                     padding='same'  )#, normalizer_fn=layers.batch_norm)  # 1/16
                #net4 = slim.max_pool2d(net4, [2, 2])
            with tf.name_scope('DS_5'):
                net5 = slim.conv2d(net4, 512, [2, 2], 2,
                                     padding='same'  )#, normalizer_fn=layers.batch_norm)  # 1/16
                #net5 = slim.max_pool2d(net5, [2, 2])

            with tf.name_scope('conv_1'):
                net = res_block(net5, 512, 512, 'res_1')

            with tf.name_scope('conv_2'):
                net = res_block(net, 512, 512, 'res_2')

            with tf.name_scope('conv_3'):
                net = res_block(net, 512, 512, 'res_3')
            with tf.name_scope('conv_4'):
                net = res_block(net, 512, 512, 'res_4')
            '''
            with tf.name_scope('conv_7'):
                net_1, net_2 = incp(net, net, 2, 5, 1, 1, 128)
                net = tf.concat([net_1, net_2], 3)
                net = layers.conv2d(net, 128, [1, 1], 1, padding='same', normalizer_fn=layers.batch_norm)
            '''
            with tf.name_scope('deconv_1'):
                net = slim.conv2d_transpose(net, 256, [2, 2], 2, padding='same', normalizer_fn=slim.batch_norm)

                #net = tf.concat([net, net4], -1)

            with tf.name_scope('deconv_2'):
                net = slim.conv2d_transpose(net, 128, [2, 2], 2, padding='same', normalizer_fn=slim.batch_norm)
                #net33=slim.conv2d(net3,128,[3,3],1)
                #net = tf.concat([net, net3], -1)

            with tf.name_scope('deconv_3'):
                net = slim.conv2d_transpose(net, 32, [2, 2], 2, padding='same', normalizer_fn=slim.batch_norm)
                #net22 = slim.conv2d(net2, 32, [3, 3], 1)
                #net = tf.concat([net, net2], -1)

            with tf.name_scope('deconv_3'):
                net = slim.conv2d_transpose(net, 16, [2, 2], 2, padding='same', normalizer_fn=slim.batch_norm)

                #net=tf.concat([net,net1],-1)

            with tf.name_scope('deconv_4'):
                net = slim.conv2d_transpose(net, 4, [2, 2], 2, padding='same', normalizer_fn=slim.batch_norm)



            with tf.name_scope('predict'):
                predict = slim.conv2d(net, 1, [1, 1], 1, activation_fn=tf.nn.sigmoid,normalizer_fn=slim.batch_norm)
                predict = tf.identity(predict, name='predicted')
                tf.summary.image('predict', predict, 1)

        return x,y,predict,net1,net2,net3,net4,net5

    def train(self,epoch=20,batch_size=5):

        x_size = 256
        y_size = 256

        w_l=tf.placeholder(tf.float32,[5])
        w_l_1=w_l[0]
        w_l_2 = w_l[1]
        w_l_3 = w_l[2]
        w_l_4 = w_l[3]
        w_l_5 = w_l[4]

        w_pp = tf.placeholder(tf.float32, [1])
        w_p=w_pp[0]




        def _parser(example_proto):
            # reading from each example of f record
            # we should write the keys here to read from each line and put in right places
            features = {  # 'height': tf.FixedLenFeature((), tf.int64),
                # 'width': tf.FixedLenFeature((), tf.int64),
                # 'depth': tf.FixedLenFeature((), tf.int64),
                'shape_i': tf.FixedLenFeature((), tf.string),
                'shape_l': tf.FixedLenFeature((), tf.string),

                'label_raw': tf.FixedLenFeature((), tf.string),
                'image_raw': tf.FixedLenFeature((), tf.string)}

            parsed_features = tf.parse_single_example(example_proto, features)

            # extract image and its size
            image_raw = parsed_features["image_raw"]
            # height = parsed_features["height"]
            # width = parsed_features["width"]
            # depth = parsed_features["depth"]
            shape_i = parsed_features["shape_i"]
            shape_l = parsed_features["shape_l"]
            label_raw = parsed_features["label_raw"]

            # some pre processing to make image an array
            # used to mread byte type data
            shape_l = tf.decode_raw(shape_l, tf.int32)
            shape_i = tf.decode_raw(shape_i, tf.int32)
            image_1d = tf.decode_raw(image_raw,
                                     tf.uint8)  # that image is now in one line  we need to reshape it later with the saved dimentions
            label_1d = tf.decode_raw(label_raw,
                                     tf.uint8)  # that image is now in one line  we need to reshape it later with the saved dimentions
            # image_shape = [height, width, depth]

            image = tf.reshape(image_1d, shape_i)
            label = tf.reshape(label_1d, shape_l)
            # label=tf.concat([label,tf.zeros(shape_l,tf.uint8),tf.zeros(shape_l,tf.uint8)],2)
            image_pros = tf.image.resize_image_with_crop_or_pad(image, target_height=y_size, target_width=x_size)
            label_pros = tf.image.resize_image_with_crop_or_pad(label, target_height=y_size, target_width=x_size)
            label_pros = tf.divide(label_pros, 255)
            # image=tf.reshape(image_1d ,[height,width,depth])

            return image_pros, label_pros

        filename = "e:\Archive\HKU-IS\\tfr\\train.tfrecords"  # shere the tf record is saved
        #batch_size = 5  # 4
        shuffle = True
        repeat = None

        global_step=0
        dataset = tf.contrib.data.TFRecordDataset(filename)  # saying we want dataset that reads from tfrecord
        dataset = dataset.map(_parser)  # pass to parser to do a little reading and processind
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=5)
        iterator = dataset.make_one_shot_iterator()
        # here we say get the next data and pass it
        # we use this object to read our data
        next_element = iterator.get_next()

        data_size = 4447
        num_to_epoch = int(data_size / batch_size)
        it = epoch * num_to_epoch
        print('iter to do ', it)




        #*************************************************************************

        x, y, predict, net1, net2, net3, net4,net5=self.create()

        with tf.variable_scope('vgg_16', 'vgg_16') as sc:
            vgg_img_5, vgg_img_4, vgg_img_3, vgg_img_2, vgg_img_1 = vgg_16(x)

        with tf.variable_scope('loss'):
            learning_rate = tf.placeholder(tf.float32, None)

            #################################################################################################################
            #'''
            with tf.name_scope('learning_rate'):
                global_step = tf.Variable(1, dtype=tf.int64)
                initial_rate = .001
                decay_steps = it
                decay_rate = .01
                learning_rate = tf.train.exponential_decay(initial_rate, global_step=global_step,
                                                           decay_steps=decay_steps,
                                                           decay_rate=decay_rate, name='learning_rate')
            #'''


            ###############################################################################################################

            with tf.name_scope('vgg_decoder'):
                '''
                f_loss_5 = tf.reduce_mean(tf.norm(vgg_img_5 - net5, ord='euclidean', axis=[-2, -1]))
                f_loss_4 = tf.reduce_mean(tf.norm(vgg_img_4 - net4, ord='euclidean', axis=[-2, -1]))
                f_loss_3 = tf.reduce_mean(tf.norm(vgg_img_3 - net3, ord='euclidean', axis=[-2, -1]))
                f_loss_2 = tf.reduce_mean(tf.norm(vgg_img_2 - net2, ord='euclidean', axis=[-2, -1]))
                f_loss_1 = tf.reduce_mean(tf.norm(vgg_img_1 - net1, ord='euclidean', axis=[-2, -1]))
                '''
                f_loss_5 = tf.reduce_mean(tf.losses.absolute_difference(labels=vgg_img_5 ,predictions= net5))
                f_loss_4 = tf.reduce_mean(tf.losses.absolute_difference(labels=vgg_img_4 ,predictions= net4))
                f_loss_3 = tf.reduce_mean(tf.losses.absolute_difference(labels=vgg_img_3 ,predictions= net3))
                f_loss_2 = tf.reduce_mean(tf.losses.absolute_difference(labels=vgg_img_2 ,predictions= net2))
                f_loss_1 =tf.reduce_mean(tf.losses.absolute_difference(labels=vgg_img_1 ,predictions= net1))

                f_loss = w_l_3*f_loss_3 + w_l_2*f_loss_2 + w_l_5*f_loss_5 + w_l_4*f_loss_4+w_l_1*f_loss_1

                tf.summary.scalar('loss_feature_5', f_loss_5)
                tf.summary.scalar('loss_feature_4', f_loss_4)
                tf.summary.scalar('loss_feature_3', f_loss_3)
                tf.summary.scalar('loss_feature_2', f_loss_2)
                tf.summary.scalar('loss_feature_1', f_loss_1)

                tf.summary.scalar('loss_feature', f_loss)

            '''        
            with tf.name_scope('style_loss'):
                #[?, 72, 72, 128]  50
                #[?, 36, 36, 512]  12
                #[?,1,1,4096]3
                s_img_g_5 = gram(s_img_5)  # , c=4096, h=3, w=3)
                s_img_g_4 = gram(s_img_4)  # , c=4096, h=3, w=3)
                s_img_g_3 = gram(s_img_3)#, c=4096, h=3, w=3)
                print(s_img_g_3)
                s_img_g_2 = gram(s_img_2)#, c=512, h=36, w=36)
                print(s_img_g_2)
                s_img_g_1 = gram(s_img_1)#, c=128, h=144, w=144)
                print(s_img_g_1)

                new_img_g_5 = gram(new_s_5)
                new_img_g_4 = gram(new_s_4)  # , c=4096, h=3, w=3)
                new_img_g_3 = gram(new_s_3)#, c=4096, h=3, w=3)
                print(new_img_g_3)
                new_img_g_2 = gram(new_s_2)#, c=512, h=36, w=36)
                print(new_img_g_2)
                new_img_g_1 = gram(new_s_1)#, c=128, h=144, w=144)
                print(new_img_g_1)


                l_s_1=tf.reduce_mean(tf.norm(new_img_g_1-s_img_g_1,ord='fro',axis=[-2,-1]))
                l_s_2 = tf.reduce_mean(tf.norm(new_img_g_2 - s_img_g_2,ord='fro',axis=[-2,-1]))
                l_s_3 =tf.reduce_mean(tf.norm(new_img_g_3 - s_img_g_3,ord='fro',axis=[-2,-1]))
                l_s_4 = tf.reduce_mean(tf.norm(new_img_g_4 - s_img_g_4, ord='fro', axis=[-2, -1]))
                l_s_5 = tf.reduce_mean(tf.norm(new_img_g_5 - s_img_g_5, ord='fro', axis=[-2, -1]))
                tf.summary.scalar('loss_style_1', l_s_1)
                tf.summary.scalar('loss_style_2', l_s_2)
                tf.summary.scalar('loss_style_3', l_s_3)
                tf.summary.scalar('loss_style_4', l_s_4)
                tf.summary.scalar('loss_style_5', l_s_5)


                s_loss = (1 * l_s_1 +1 * l_s_2 +1* l_s_3 + 1 * l_s_4+ 1 * l_s_5) /batch_size
                #s_loss=l_s_4
                tf.summary.scalar('S_loss', s_loss)

            '''
            with tf.name_scope('pixel_loss'):
                #pix_loss = tf.norm(predict - y, ord='euclidean')
                #pix_loss=tf.losses.absolute_difference(labels=y,predictions=predict)
                pix_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=predict)
                pix_loss=tf.reduce_mean(pix_loss)
                print('pix',pix_loss)

                tf.summary.scalar('loss_pixel', pix_loss)

                # loss =f_loss
                loss = 1 * f_loss + w_p * pix_loss  # * tv_loss + 1 * s_loss#+10 * pix_loss +

            loss = tf.identity(loss, name='loss')

            tf.summary.scalar('loss', loss)

            with tf.name_scope('optimizer'):
                '''
                update_ops1 = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope='loss_opti')
                update_ops2 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope= 'trans_net')
                update_ops=update_ops1+update_ops2
                '''
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                           name='optimizer')
                global_step = global_step + 1
        sess = tf.Session()

        # here we load weights from ckpt provided in slim

        colect_vgg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_16')
        saver1 = tf.train.Saver(var_list=colect_vgg)
        # print(tf.get_default_graph().get_operations())
        saver1.restore(sess, "e:\Archive\pretrained nets\slim\\vgg16\\vgg_16.ckpt")  # address of provide checkpoint
        print('restored baby !!!! ')

        # ********************************************************************

        # *****************************************************************************************************************************
        colect1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope='trans_net')  # GET A collection of variabels in salency_net
        colect2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='loss')
        colect = colect1 + colect2

        init = tf.variables_initializer(colect)  # initialize the collection
        sess.run(init)

        summary_writer = tf.summary.FileWriter("tensorboard\\", tf.get_default_graph(), max_queue=20)  # for tensorboard
        summary_writer.add_graph(tf.get_default_graph())  # for tensorboard
        merged = tf.summary.merge_all()
        saver = tf.train.Saver(var_list=colect1)

        # ************************************

        # c=(b,b,b,b)

        # b=np.concatenate(c,0)

        for itter in range(it):
            a = sess.run(next_element)
            if itter < 2000:
                W_L=[1,1,1,1,1]
                W_P=5000
                #L=1e-3
            else:
                W_L = [0, 0, 0, 0, 0]
                W_P = 5000
                #L = 1e-3



            summary, _ = sess.run([merged, optimizer], feed_dict={x: a[0], y: a[1],w_l:W_L,w_pp:[W_P]})#,learning_rate:L})



            if itter % 200 == 0:
                print(W_L)
                print(W_P)
                summary_writer.add_summary(summary, itter)
                saver.save(sess, "model\\model")

        summary_writer.add_summary(summary, itter)
        saver.save(sess, "model\\model")




    def use(self,img,model_path=None):
        tf.reset_default_graph()

        x, y, predict, net1, net2, net3, net4, net5 = self.create()

        a = Image.open(img)#'E:\Archive\image process\saliency setection\MSRA10K_Imgs_GT\MSRA10K_Imgs_GT\Imgs\\175671.jpg')
        a = a.resize((256, 256))
        a = np.array(a, np.uint8)
        a_shape = np.array(a.shape)
        a = np.reshape(a, [1, a_shape[0], a_shape[1], a_shape[2]])
        with tf.device('/cpu:0'):
            sess = tf.Session()
            #saver = tf.train.import_meta_graph(model_path)  # add the graph here
            saver=tf.train.Saver()
            if model_path == None:
                saver.restore(sess, tf.train.latest_checkpoint('model\\'))
            else:
                saver.restore(sess, tf.train.latest_checkpoint(model_path))

            start = time.time()
            Predict = sess.run(predict, feed_dict={x: a})
        end = time.time()

        #print('time:', end - start)

        Predict = Predict
        Predict = np.array(Predict )
        pred_shape = np.array(Predict.shape)
        #print(pred_shape)
        Predict = np.reshape(Predict[0], pred_shape[1:3])

        #img = Image.fromarray(Predict)
        # img=img.filter(ImageFilter.BLUR)

        #img.show()
        return Predict

    def load(self,model_path=None):
        x, y, predict, net1, net2, net3, net4, net5 = self.create()

        sess = tf.Session()
        # saver = tf.train.import_meta_graph(model_path)  # add the graph here
        saver = tf.train.Saver()
        if model_path == None:
            saver.restore(sess, tf.train.latest_checkpoint('model\\'))
        else:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))

        return sess,x, y, predict, net1, net2, net3, net4, net5

    def evaluate(self,name):
        import tensorflow as tf
        from PIL import Image
        import numpy as np
        import pickle
        import pandas as pd



        def image_post_pro(np_img):
            shape = np_img.shape
            new_img = np.zeros(np.array(shape), np.int32)
            tresh = np.mean(np.mean(np_img, -1), -1)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    if np_img[i][j] > tresh:
                        new_img[i][j] = 255

                    else:
                        new_img[i][j] = 0
            image = Image.fromarray(np_img)
            # image.show()
            return new_img


        def threed_img_pro(np_img):
            shape = np_img.shape

            batch=shape[0]
            h=shape[1]
            w=shape[2]
            new_img = np.zeros(np.array(shape), np.int32)
            tresh = np.mean(np.mean(np.mean(np_img, -1), -1),-1)
            for k in range(batch):
                for i in range(h):
                    for j in range(w):
                        if np_img[k][i][j] > tresh:
                            new_img[k][i][j] = 255

                        else:
                            new_img[k][i][j] = 0

            # image.show()
            return new_img



        def evaluate(gt, m, b2=.3):

            shape = np.array(gt.shape)

            m_and_g = 0
            G = 0
            M = 0
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if m[i, j] == 255:
                        M = M + 1

                    if gt[i, j] == 255:
                        G = G + 1
                        if gt[i, j] == m[i, j]:
                            m_and_g = m_and_g + 1
            if M == 0:
                M = 1
            if G == 0:
                G = 1
            percision = m_and_g / M
            recall = m_and_g / G

            MEA = np.mean(np.abs(gt - m) / 255)

            n = b2 * percision + recall
            if n == 0:
                n = .01

            f_measure = (1 + b2) * percision * recall / n

            return percision, recall, f_measure, MEA

        def evaluate_3d(gt, m, b2=.3):

            shape = np.array(gt.shape)

            m_and_g = 0
            G = 0
            M = 0
            for k in range(shape[0]):
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        if m[k,i, j] == 255:
                            M = M + 1

                        if gt[k,i, j] == 255:
                            G = G + 1
                            if gt[k,i, j] == m[k,i, j]:
                                m_and_g = m_and_g + 1
            if M == 0:
                M = 1
            if G == 0:
                G = 1
            percision = m_and_g / M
            recall = m_and_g / G

            MEA = np.mean(np.abs(gt - m) / 255)

            n = b2 * percision + recall
            if n == 0:
                n = .01

            f_measure = (1 + b2) * percision * recall / n

            return percision, recall, f_measure, MEA

        # *******************************************************************************************************************************
        x_size = 256
        y_size = 256

        h = 256
        w = 256

        def _parser(example_proto):
            # reading from each example of f record
            # we should write the keys here to read from each line and put in right places
            features = {  # 'height': tf.FixedLenFeature((), tf.int64),
                # 'width': tf.FixedLenFeature((), tf.int64),
                # 'depth': tf.FixedLenFeature((), tf.int64),
                'img_shape': tf.FixedLenFeature((), tf.string),
                'label_shape': tf.FixedLenFeature((), tf.string),

                'label_raw': tf.FixedLenFeature((), tf.string),
                'img_raw': tf.FixedLenFeature((), tf.string)}

            parsed_features = tf.parse_single_example(example_proto, features)

            # extract image and its size
            image_raw = parsed_features["img_raw"]
            # height = parsed_features["height"]
            # width = parsed_features["width"]
            # depth = parsed_features["depth"]
            shape_i = parsed_features["img_shape"]
            shape_l = parsed_features["label_shape"]
            label_raw = parsed_features["label_raw"]

            # some pre processing to make image an array
            # used to mread byte type data
            shape_l = tf.decode_raw(shape_l, tf.int32)
            shape_i = tf.decode_raw(shape_i, tf.int32)
            image_1d = tf.decode_raw(image_raw,
                                     tf.uint8)  # that image is now in one line  we need to reshape it later with the saved dimentions
            label_1d = tf.decode_raw(label_raw,
                                     tf.uint8)  # that image is now in one line  we need to reshape it later with the saved dimentions
            # image_shape = [height, width, depth]

            image = tf.reshape(image_1d, shape_i)
            label = tf.reshape(label_1d, shape_l)
            image_pros = tf.image.resize_image_with_crop_or_pad(image, target_height=y_size, target_width=x_size)
            label_pros = tf.image.resize_image_with_crop_or_pad(label, target_height=y_size, target_width=x_size)
            label_pros = tf.divide(label_pros, 255)
            # image=tf.reshape(image_1d ,[height,width,depth])

            return image_pros, label_pros

        # the parameters
        # filename = "D:\Archive\AI\image process\saliency\HKU-IS\\tfr\\train.tfrecords" # where the tf record is saved
        filename = "E:\Archive\image process\saliency setection\MSRA10K_Imgs_GT\MSRA10K_Imgs_GT\\tfr\\train_with_map.tfrecords"  # where the tf record is saved
        batch_size = 20
        shuffle = False
        repeat = None

        dataset = tf.data.TFRecordDataset(filename)  # saying we want dataset that reads from tfrecord
        dataset = dataset.map(_parser)  # pass to parser to do a little reading and processind
        #dataset = dataset.repeat(repeat)
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=200)

        # making iterator for feeding
        # iterator gets data from dataset object
        iterator = dataset.make_one_shot_iterator()
        # here we say get the next data and pass it
        # we use this object to read our data
        next_element = iterator.get_next()

        sess, x, y, predict, net1, net2, net3, net4, net5=self.load()



        all_percision = []
        all_recall = []
        all_fmeasure = []
        all_MEA = []

        l = 10000
        data = []

        for num in range(int(l/batch_size)):

            a = sess.run(next_element)

            Predict = sess.run(predict, feed_dict={x: a[0]})
            Predict = Predict * 255

            Predict = np.array(Predict, np.uint8)
            pred_shape = np.array(Predict.shape)

            Predict = np.reshape(Predict, pred_shape[0:3])
            # ************************************************************************************************************
            bin_predict = threed_img_pro(Predict)

            gt = np.array(a[1], np.uint8)
            # print(gt.shape)

            gt = np.reshape(gt, [batch_size,h, w])

            percision, recall, f_measure, MEA = evaluate_3d(gt * 255, bin_predict)

            if num == 0:
                '''
                Image.fromarray(gt * 255).show()
                Image.fromarray(Predict).show()
                '''
                print(percision, recall, f_measure, MEA)

            all_fmeasure.append(f_measure)
            all_MEA.append(MEA)
            all_percision.append(percision)
            all_recall.append(recall)
            data.append([f_measure, MEA, percision, recall])

        w_MEA = sum(all_MEA) / l
        w_f_measure = sum(all_fmeasure) / l
        w_percision = sum(all_percision) / l
        w_recall = sum(all_recall) / l

        print(w_f_measure, w_MEA, w_percision, w_recall)

        frame = pd.DataFrame(data, columns=['f_meas', 'MEA', 'percision', 'recall'])
        frame.to_csv(name+'.csv')


    ############################################
    ############################################
    def load_freez(self, model_path=None):
        tf.reset_default_graph()

        with tf.device("/cpu:0"):
            with tf.Graph().as_default() as load_graph:

                sess = tf.Session(graph=load_graph)
                # with tf.Session(graph=load_graph) as sess:
                x,y,predict,net1,net2,net3,net4,net5 = self.create()

                # col=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sal_net')
                saver = tf.train.Saver()
                if model_path == None:
                    saver.restore(sess, tf.train.latest_checkpoint('model\\'))
                else:
                    saver.restore(sess, tf.train.latest_checkpoint(model_path))

                return load_graph, sess

    def freez_pre_trained(self, model_path='model\\', checkpoint_path='model', out_file='model\model.pb',
                          predict_string=None):
        tf.reset_default_graph()

        load_graph, sess = self.load_freez(model_path)


        if predict_string is None:
            predict_string = self.predict_node

        print(len(load_graph.get_operations()))
        print(load_graph.get_operations())
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, load_graph.as_graph_def(),
                                                                        [predict_string], )
        describe_graph(load_graph.as_graph_def(), False)
        describe_graph(output_graph_def, True)
        with tf.gfile.GFile(out_file, "wb") as f:
            f.write(output_graph_def.SerializeToString())
            f.close()
        print("freezing and dumping is done baby !!! :D")

    def freez(self, model_path='model\model.meta', checkpoint_path='model', out_file='model\model.pb',
              predict_string=None):
        '''

        :param model_path: path to model .meta file
        :param checkpoint_path: path of checkpoints
        :param out_path: some .pb file
        :return:
        '''
        tf.reset_default_graph()
        if predict_string is None:
            predict_string = self.predict_node

        clear_devices = True
        load_graph = tf.Graph()
        with tf.Session(graph=load_graph) as sess:
            saver = tf.train.import_meta_graph(model_path)  # , clear_devices=clear_devices)
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

            print(load_graph.get_operations())
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, load_graph.as_graph_def(),
                                                                            [predict_string], )
        describe_graph(load_graph.as_graph_def(), False)
        describe_graph(output_graph_def, True)
        with tf.gfile.GFile(out_file, "wb") as f:
            f.write(output_graph_def.SerializeToString())
            f.close()
        print("freezing and dumping is done baby !!! :D")

    def load_graph(self, frozen_graph_filename='model\model.pb'):
        tf.reset_default_graph()

        with tf.device("/cpu:0"):
            # We load the protobuf file from the disk and parse it to retrieve the
            # unserialized graph_def
            with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # Then, we import the graph_def into a new Graph and returns it
            with tf.Graph().as_default() as graph:
                # The name var will prefix every op/nodes in your graph
                # Since we load everything in a new graph, this is not needed
                tf.import_graph_def(graph_def, name="")
            print(len(graph.get_operations()))
            # print([n.name for n in graph.as_graph_def().node])
            self.graph = graph

        return graph

    def sess_from_graph(self, input_str=None, output_str=None):

        if input_str is None:
            input_str = self.input_node
        if output_str is None:
            output_str = self.predict_node
        graph = self.graph
        sess = tf.Session(graph=graph)
        x = graph.get_tensor_by_name(input_str)
        predict = graph.get_tensor_by_name(output_str)
        self.sess = sess
        self.input = x
        self.predict = predict
        return sess, x, predict

        # print(graph.get_operations())

    def optimize_graph(self, graph_filename='model/model.pb', output_node=None, out_folder='model'):
        '''

        :param model_dir:
        :param graph_filename:
        :param transforms:
         [‘remove_nodes(op=Identity)’,
     ‘merge_duplicate_nodes’,
     ‘strip_unused_nodes’,
     ‘fold_constants(ignore_errors=true)’,
     ‘fold_batch_norms’,
     ‘quantize_nodes’, # Quantization
     ‘quantize_weights’, # Quantization

     ]

        :param output_node:
        :return:
        '''
        transforms = ['remove_nodes(op=Identity)',
                      # 'merge_duplicate_nodes',
                      'strip_unused_nodes',
                      'fold_constants(ignore_errors=true)',
                      'fold_batch_norms',
                      # 'quantize_nodes', # Quantization
                      # 'quantize_weights', # Quantization

                      ]
        tf.reset_default_graph()
        if output_node is None:
            output_node = self.predict_node
        input_names = []
        output_names = [output_node]
        graph_def = self.load_graph(graph_filename)

        graph_def = graph_def.as_graph_def()
        optimized_graph_def = TransformGraph(
            graph_def,
            input_names,
            output_names,
            transforms)
        describe_graph(optimized_graph_def, True)
        tf.train.write_graph(optimized_graph_def,
                             logdir=out_folder,
                             as_text=False,
                             name='optimized_model.pb')
        print('Graph optimized!')

    def use_pb(self, img_dir):
        img = Image.open(img_dir)
        img = img.convert('RGB')
        img = img.resize((256, 256))
        img = np.array(img)
        height = img.shape[0]
        width = img.shape[1]
        img = np.array(img).reshape([1, height, width, 3])

        sess = self.sess
        x = self.input
        predict = self.predict
        with tf.device("/cpu:0"):
            s = time.time()
            out = sess.run(predict, feed_dict={x: img})
            e = time.time()
            print('exec time', e - s)
        out = np.array(out)
        out_shape = out.shape
        print(out_shape)

        out = np.reshape(out, [out_shape[1], out_shape[2]])
        return out,e-s







