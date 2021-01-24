
import tensorflow as tf
from PIL import Image
from tensorflow.contrib import layers ,slim
import numpy as np
import os
import time
from tensorflow.tools.graph_transforms import TransformGraph

def atrous(x,depth,size,rate,name_var='atrous',padding='SAME',activation_fn=None,normalizer_fn=None):
    '''
    makes atrous convolution with obtional normalizer function
    :param x:
    :param depth:
    :param size:
    :param rate:
    :param name_var:
    :param padding:
    :param normalizer_fn:
    :return:
    '''
    in_shape=x.shape[-1]
    filter=tf.get_variable(name_var,[size[0],size[1],in_shape,depth],tf.float32,slim.xavier_initializer())
    net=tf.nn.atrous_conv2d(x,filter,rate,padding=padding)
    if normalizer_fn is not None:
        net=activation_fn(net)
    if normalizer_fn is not None:
        net=normalizer_fn(net)
    return net


def res_block(x,depth,size,normalizer_fn=None):
    init_d=x.shape[-1]

    net=slim.conv2d(x,depth,size,1,normalizer_fn=normalizer_fn)


    net = slim.conv2d(net, init_d, [1,1], 1, normalizer_fn=normalizer_fn)
    net=net+x
    return net

def res_atrous_block(x,depth,size,normalizer_fn):
    init_d=x.shape[-1]
    net=slim.conv2d(x,depth,size,1,normalizer_fn=normalizer_fn)
    net=slim.conv2d(net,depth,size,1,normalizer_fn=normalizer_fn)
    net = slim.conv2d(net, init_d, size, 1, normalizer_fn=normalizer_fn)
    net=net+x
    return net
def batch_norm(x,is_training=False):
    return slim.batch_norm(x,is_training=is_training)


def incp(x_1,x_2,f_1,f_2,s_1,s_2,dep):
    #incp(x_1, x_2, x_3, f_1, f_2, f_3, s_1, s_2, s_3, dep)
    net_1 = layers.conv2d(x_1, dep, [f_1, f_1], s_1, padding='same',normalizer_fn=layers.batch_norm)
    net_2 = layers.conv2d(x_2, dep, [f_2, f_2], s_2, padding='same',normalizer_fn=layers.batch_norm)
    #net_3 = layers.conv2d(x_3, dep, [f_3, f_3], s_3, padding='same', normalizer_fn=layers.batch_norm)


    return net_1 , net_2#,net_3



def pyramid(x,d,s1,r1,s2,r2,s3,r3,name='pyramid_',activation_fn=None,normalizer_fn=None):
    n1=atrous(x,d,s1,r1,name+'atrous_'+'1',activation_fn=activation_fn,normalizer_fn=normalizer_fn)
    n2 = atrous(x, d, s2, r2, name + 'atrous_' + '2', activation_fn=activation_fn, normalizer_fn=normalizer_fn)
    n3 = atrous(x, d, s3, r3, name + 'atrous_' + '3', activation_fn=activation_fn, normalizer_fn=normalizer_fn)

    net=tf.concat((n1,n2,n3),-1)
    return net

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

#***************************************************************************************


class AtrousNet:

    def create(self,is_training=False):
        bn=lambda x:batch_norm(x,is_training)
        x=tf.placeholder(tf.float32, [None,None,None,3])


        with tf.variable_scope('sal_net'):
            net = slim.conv2d(x, 32, [3, 3], 4, normalizer_fn=bn)  # 1/4

            net = res_block(net, 32, [3, 3], bn)
            net = res_block(net, 32, [3, 3], bn)
            net = slim.conv2d(net, 64, [3, 3], 2, normalizer_fn=bn)  # 1/8

            net = res_block(net, 64, [3, 3], bn)
            net = res_block(net, 64, [3, 3], bn)
            net = slim.conv2d(net, 128, [3, 3], 2, normalizer_fn=bn)  # 1/16

            net = pyramid(net, 128, [3, 3], 2, [5, 5], 3, [9, 9], 4, name='pyramid1', activation_fn=tf.nn.relu,
                          normalizer_fn=bn)

            net = slim.conv2d_transpose(net, 256, [2, 2], 2, normalizer_fn=bn)  # 1/8
            net = slim.conv2d_transpose(net, 128, [2, 2], 2, normalizer_fn=bn)  # 1/4
            net = slim.conv2d_transpose(net, 32, [2, 2], 2, normalizer_fn=bn)  # 1/2
            net = slim.conv2d_transpose(net, 16, [2, 2], 2, normalizer_fn=bn)  # 1/1

            net = slim.conv2d(net, 1, [1, 1], activation_fn=tf.sigmoid)

        return net,x




    ############################################################################################################################

    def train(self,epoch,batch_size=20):
        #handling data

        data_size = 4447
        num_to_epoch = int(data_size / batch_size)
        it = epoch * num_to_epoch
        def _parser(example_proto):
            i_size = 256
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
            image_pros = tf.image.resize_image_with_crop_or_pad(image, target_height=i_size, target_width=i_size)
            label_pros = tf.image.resize_image_with_crop_or_pad(label, target_height=i_size, target_width=i_size)
            label_pros = tf.divide(label_pros, 255)
            # image=tf.reshape(image_1d ,[height,width,depth])

            return image_pros, label_pros

        # the parameters
        filename = "E:\Archive\HKU-IS\\tfr\\train.tfrecords"  # shere the tf record is saved


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
            learning_rate = tf.train.exponential_decay(initial_rate, global_step=global_step, decay_steps=decay_steps,
                                                       decay_rate=decay_rate, name='learning_rate')

        ###############################################################################################################
        y_ = tf.placeholder(tf.float32, [None, None, None, 1])
        tf.summary.image('label', y_)
        predict,x=self.create(is_training=True)
        tf.summary.image('input', x)
        tf.summary.image('predict', predict)

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
                optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step,
                                                                           name='optimizer')
                global_step = global_step + 1

        tf.summary.scalar('learning_rate', learning_rate)

        summary_writer = tf.summary.FileWriter("tensorboard\\", tf.get_default_graph())  # for tensorboard

        variable_init = tf.global_variables_initializer()  # initializing variables
        save_colection=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sal_net')
        saver = tf.train.Saver(var_list=save_colection)  # for saving weights and the model

        summary_writer.add_graph(tf.get_default_graph())  # for tensorboard

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True

        # sess=tf.Session(config=config)
        sess = tf.Session()
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess) # for debug and see variables
        sess.run(variable_init)
        merged = tf.summary.merge_all()

        for itter in range(it):
            a = sess.run(next_element)
            summary, _ = sess.run([merged, optimizer], feed_dict={x: a[0], y_: a[1]})

            if itter % 200 == 0:
                summary_writer.add_summary(summary, itter)
                saver.save(sess, "model\\model")

        summary_writer.add_summary(summary, itter)
        saver.save(sess, "model\\model")

#######################################################################################################################

    def use(self,img_dir,model_path=None):
        tf.reset_default_graph()
        img=Image.open(img_dir)
        img=img.resize((256,256))
        img=np.array(img)
        height=img.shape[0]
        width=img.shape[1]
        img=np.array(img).reshape([1,height,width,3])
        with tf.device("/cpu:0"):
            predict, x, sess=self.load(model_path)
            out=sess.run(predict,feed_dict={x:img})
        out=np.array(out)
        out_shape=out.shape
        out=np.reshape(out,[out_shape[1],out_shape[2]])
        return out

    def load(self,model_path=None):
        with tf.device("/cpu:0"):
            predict,x=self.create()
            sess=tf.Session()
            saver=tf.train.Saver()
            if model_path==None:
                saver.restore(sess,tf.train.latest_checkpoint('model\\'))
            else:
                saver.restore(sess, tf.train.latest_checkpoint(model_path))

        return predict,x,sess

    def load_freez(self, model_path=None):
        tf.reset_default_graph()

        with tf.device("/cpu:0"):
            with tf.Graph().as_default() as load_graph:

                sess = tf.Session(graph=load_graph)
                # with tf.Session(graph=load_graph) as sess:
                predict, x = self.create(is_training=False)

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

        out = np.reshape(out, [out_shape[1], out_shape[2]])
        return out,e-s



    def evaluate(self, name):
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

            batch = shape[0]
            h = shape[1]
            w = shape[2]
            new_img = np.zeros(np.array(shape), np.int32)
            tresh = np.mean(np.mean(np.mean(np_img, -1), -1), -1)
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
                        if m[k, i, j] == 255:
                            M = M + 1

                        if gt[k, i, j] == 255:
                            G = G + 1
                            if gt[k, i, j] == m[k, i, j]:
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
        # dataset = dataset.repeat(repeat)
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=5)

        # making iterator for feeding
        # iterator gets data from dataset object
        iterator = dataset.make_one_shot_iterator()
        # here we say get the next data and pass it
        # we use this object to read our data
        next_element = iterator.get_next()

        predict, x, y_, sess = self.load()

        all_percision = []
        all_recall = []
        all_fmeasure = []
        all_MEA = []

        l = 10000
        data = []

        for num in range(int(l / batch_size)):

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

            gt = np.reshape(gt, [batch_size, h, w])

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
        frame.to_csv(name + '.csv')


