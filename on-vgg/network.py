import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib import layers
from PIL import Image
import numpy as np
import pandas as pd


def incp(x_1, x_2, f_1, f_2, s_1, s_2, dep):
    # incp(x_1, x_2, x_3, f_1, f_2, f_3, s_1, s_2, s_3, dep)
    net_1 = layers.conv2d(x_1, dep, [f_1, f_1], s_1, padding='same', normalizer_fn=layers.batch_norm)
    net_2 = layers.conv2d(x_2, dep, [f_2, f_2], s_2, padding='same', normalizer_fn=layers.batch_norm)
    # net_3 = layers.conv2d(x_3, dep, [f_3, f_3], s_3, padding='same', normalizer_fn=layers.batch_norm)
    return net_1, net_2  # ,net_3

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


class Network:

    def incp(x_1, x_2, f_1, f_2, s_1, s_2, dep):
        # incp(x_1, x_2, x_3, f_1, f_2, f_3, s_1, s_2, s_3, dep)
        net_1 = layers.conv2d(x_1, dep, [f_1, f_1], s_1, padding='same', normalizer_fn=layers.batch_norm)
        net_2 = layers.conv2d(x_2, dep, [f_2, f_2], s_2, padding='same', normalizer_fn=layers.batch_norm)
        # net_3 = layers.conv2d(x_3, dep, [f_3, f_3], s_3, padding='same', normalizer_fn=layers.batch_norm)

    def create(self):

        # inputs

        with tf.name_scope('input'):
            x = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='input')
            tf.summary.image('input', x, 1)
        with tf.variable_scope('vgg_16', 'vgg_16') as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(x, 2, slim.conv2d, 64, [3, 3], scope='conv1')  # 300
                net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 150
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 75
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 37
                print(net)
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                print(net)
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')
            print(net)

        # concating inputs to for four channel data

        sess = tf.Session()
        saver = tf.train.Saver()
        # print(tf.get_default_graph().get_operations())
        saver.restore(sess, "e:\Archive\pretrained nets\slim\\vgg16\\vgg_16.ckpt")  # address of provide checkpoint
        #print('restored baby !!!! ')
        # ********************************************************************

        # layers
        with tf.variable_scope('sal_net'):
            with tf.name_scope('conv_3'):
                net = res_block(net, 512, 512, 'res_1')

            with tf.name_scope('conv_4'):
                net = res_block(net, 512, 512, 'res_2')

            with tf.name_scope('conv_5'):
                net = res_block(net, 512, 512, 'res_3')
            with tf.name_scope('conv_6'):
                net = res_block(net, 512, 512, 'res_4')

            # upsampling
            net = slim.conv2d_transpose(net, 256, [2, 2], 2, normalizer_fn=slim.batch_norm, padding='valid')

            net = slim.conv2d_transpose(net, 128, [2, 2], 2, normalizer_fn=slim.batch_norm)

            net = slim.conv2d_transpose(net, 32, [2, 2], 2, normalizer_fn=slim.batch_norm)

            net = slim.conv2d_transpose(net, 16, [2, 2], 2, normalizer_fn=slim.batch_norm)
            net = slim.conv2d_transpose(net, 4, [2, 2], 2, normalizer_fn=slim.batch_norm)

            net = slim.conv2d(net, 1, [1, 1], 1, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.sigmoid)
            print(net)
            with tf.name_scope('label'):
                y_ = tf.placeholder(tf.int8, [None, None, None, 1], name='label')

            with tf.name_scope('predict'):
                predict = tf.identity(net, name='predict')

                tf.summary.image('predict', predict, 1)

        return predict, x ,y_,sess

    def train(self, epoch,batch_size=20):
        # handling data

        data_size = 4447
        num_to_epoch = int(data_size / batch_size)
        it = epoch * num_to_epoch
        x_size = 320
        y_size = 320

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
            image_pros = tf.image.resize_image_with_crop_or_pad(image, target_height=y_size, target_width=x_size)
            label_pros = tf.image.resize_image_with_crop_or_pad(label, target_height=y_size, target_width=x_size)
            label_pros = tf.divide(label_pros, 255)
            # image=tf.reshape(image_1d ,[height,width,depth])

            return image_pros, label_pros

        # the parameters
        filename = "E:\Archive\HKU-IS\\tfr\\train_with_map.tfrecords"  # shere the tf record is saved
        batch_size = 20
        shuffle = False
        repeat = None

        dataset = tf.contrib.data.TFRecordDataset(filename)  # saying we want dataset that reads from tfrecord
        dataset = dataset.map(_parser)  # pass to parser to do a little reading and processind
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=5)

        # making iterator for feeding
        # iterator gets data from dataset object
        iterator = dataset.make_one_shot_iterator()
        # here we say get the next data and pass it
        # we use this object to read our data
        next_element = iterator.get_next()

        #################################################################################################################
        with tf.name_scope('learning_rate'):
            global_step = tf.Variable(1, dtype=tf.int64)
            initial_rate = .001
            decay_steps = it
            decay_rate = .1
            learning_rate = tf.train.exponential_decay(initial_rate, global_step=global_step,
                                                       decay_steps=decay_steps,
                                                       decay_rate=decay_rate, name='learning_rate')

        ###############################################################################################################

        predict, x, y_,sess = self.create()

        ############################################################################################################
        with tf.name_scope('loss'):
            loss1=tf.losses.absolute_difference(y_,predict)
            # loss=tf.nn.l2_loss(loss1,name='loss')
            #loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=predict)

            loss = tf.reduce_mean(loss1, name='loss')

            tf.summary.scalar('loss', loss)
        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                           name='optimizer')
                global_step = global_step + 1

        tf.summary.scalar('learning_rate', learning_rate)

        summary_writer = tf.summary.FileWriter("tensorboard\\", tf.get_default_graph())  # for tensorboard

        variable_init = tf.global_variables_initializer()  # initializing variables
        save_colection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sal_net')
        saver = tf.train.Saver(var_list=save_colection)  # for saving weights and the model

        colect = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope='saliency_net')  # GET A collection of variabels in salency_net

        init = tf.variables_initializer(colect)  # initialize the collection
        sess.run(init)

        summary_writer.add_graph(tf.get_default_graph())  # for tensorboard

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True

        # sess=tf.Session(config=config)
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        for itter in range(it):
            a = sess.run(next_element)
            summary, _ = sess.run([merged, optimizer], feed_dict={x: a[0], y_: a[1]})

            if itter % 200 == 0:
                summary_writer.add_summary(summary, itter)
                saver.save(sess, "model\\model")

        summary_writer.add_summary(summary, itter)
        saver.save(sess, "model\\model")

#######################################################################################################################

    def use(self,img_dir,sal_dir):
        img=Image.open(img_dir)
        #img=img.resize((300,300))
        img=np.array(img)
        sal = Image.open(sal_dir)
        sal = np.array(sal)
        height=img.shape[0]
        width=img.shape[1]
        img=img.reshape([1,height,width,3])
        sal=sal.reshape([1,height,width,1])
        sess, predict, x, y_=self.load()
        res=sess.run(predict,feed_dict={x:img})
        res=np.array(res)
        res_shape=res.shape
        #print(res_shape)
        res=res.reshape(res_shape[1:3])
        return res

    def load(self):
        predict, x, y_ = self.create()
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('model\\'))
        return sess,predict,x,y_

    def evaluate(self):

        h=300
        w=300
        def image_post_pro(np_img):
            shape = np_img.shape
            new_img = np.zeros(np.array(shape), np.int32)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if np_img[i][j] > 128:
                        new_img[i][j] = 255

                    else:
                        new_img[i][j] = 0
            image = Image.fromarray(np_img)
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
            ##########################################################
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
            image_pros = tf.image.resize_image_with_crop_or_pad(image, target_height=h, target_width=w)
            label_pros = tf.image.resize_image_with_crop_or_pad(label, target_height=h, target_width=w)
            label_pros = tf.divide(label_pros, 255)
            # image=tf.reshape(image_1d ,[height,width,depth])

            return image_pros, label_pros

            ##################################################

        # the parameters
        filename = "E:\Archive\image process\saliency setection\MSRA10K_Imgs_GT\MSRA10K_Imgs_GT\\tfr\\train_with_map.tfrecords"  # shere the tf record is saved
        batch_size = 1
        shuffle = False
        repeat = None

        dataset = tf.contrib.data.TFRecordDataset(filename)  # saying we want dataset that reads from tfrecord
        dataset = dataset.map(_parser)  # pass to parser to do a little reading and processind
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=5)

        # making iterator for feeding
        # iterator gets data from dataset object
        iterator = dataset.make_one_shot_iterator()
        # here we say get the next data and pass it
        # we use this object to read our data
        next_element = iterator.get_next()

            ###############################################

        sess, predict, x ,y_=self.load()

        all_percision = []
        all_recall = []
        all_fmeasure = []
        all_MEA = []

        l = 10000
        data = []

        for num in range(l):
            a = sess.run(next_element)

            Predict = sess.run(predict, feed_dict={inp_rgb: a[0],inp_sal:a[2]})
            Predict = Predict * 255

            Predict = np.array(Predict, np.uint8)
            pred_shape = np.array(Predict.shape)

            Predict = np.reshape(Predict[0], pred_shape[1:3])
            # ************************************************************************************************************
            bin_predict = image_post_pro(Predict)

            gt = np.array(a[1] , np.uint8)
            #print(gt.shape)

            gt = np.reshape(gt[0], [h, w])



            percision, recall, f_measure, MEA = evaluate(gt, bin_predict)

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
        frame.to_csv('result.csv')


