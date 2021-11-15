
import os
import tensorflow as tf

from PIL import Image

import numpy as np

'''
this file handles saliency detection data sets
it gets a folder and a file type for each kind of data 
then creats a tfrecord 
'''

def write_tfr_sal(folder_rgb,folder_label,save_dir,name='train'):
    '''

    :param folder: folder containing images
    :param img_type: the type oof each image like "jpg"
    :param label_type: type of label usually "png"
    :param save_dir: where to save tfrecord
    :param name: name of tfr
    :return:
    '''
    img_names=os.listdir(folder_rgb)
    label_names=os.listdir(folder_label)

    rgb_imgs=[]
    sal_maps=[]
    n_sal_map=[]# the other maps  not GT made by dr madani code
    for i,lab in enumerate(label_names) :
        img=img_names[i]




        #if img.endswith('.'+img_type):
        rgb_imgs.append(img)
        #elif img.endswith('.'+label_type):
        sal_maps.append(lab)


    filename = os.path.join(save_dir , name + '.tfrecords')
    writer = tf.io.TFRecordWriter(filename)
    for i,img in enumerate(rgb_imgs):

        image_file=os.path.join(folder_rgb,img)
        label_file=os.path.join(folder_label,sal_maps[i])

        np_img=np.array(Image.open(image_file))
        img_shape=np.array(np_img.shape,dtype=np.int32)
        img_shape_raw=img_shape.tostring()
        np_label=np.array(Image.open(label_file))
        label_shape =np.array(np_label.shape,dtype=np.int32)
        label_shape=np.array([label_shape[0],label_shape[1],1])
        label_shape_raw = label_shape.tostring()
        serial_img=np_img.tostring()
        serial_label=np_label.tostring()


        if i == 0:
            Image.open(image_file).show()
            Image.open(label_file).show()




            print('img_shape',img_shape)
            print('label',label_shape)
            print('size',len(serial_img))



            example = tf.train.Example(features=tf.train.Features(feature={
            'img_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_shape_raw])),
            'label_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_shape_raw])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serial_img])),
            'label_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serial_label])),


            }))
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_shape_raw])),
                'label_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_shape_raw])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serial_img])),
                'label_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serial_label])),


        }))





        writer.write(example.SerializeToString())
    writer.close()

    print('done')


def _parser(proto):
    features={
        'label_raw':tf.FixedLenFeature((),tf.string),
        'img_raw': tf.FixedLenFeature((), tf.string),
        'label_shape': tf.FixedLenFeature((), tf.string),
        'img_shape': tf.FixedLenFeature((), tf.string),
        'map_raw': tf.FixedLenFeature((), tf.string),
        'map_shape': tf.FixedLenFeature((), tf.string),
    }
    parsed_features = tf.parse_single_example(proto, features)

    image_raw = parsed_features["img_raw"]
    label_raw = parsed_features["label_raw"]
    label_shape_raw = parsed_features["label_shape"]
    img_shape_raw = parsed_features["img_shape"]
    map_raw = parsed_features["map_raw"]
    map_shape_raw = parsed_features["map_shape"]

    image_raw=tf.decode_raw(image_raw,tf.uint8)
    img_shape_raw=tf.decode_raw(img_shape_raw,tf.int32)
    img_shape=tf.reshape(img_shape_raw,[3])
    label_raw = tf.decode_raw(label_raw, tf.uint8)
    label_shape_raw = tf.decode_raw(label_shape_raw, tf.int32)
    label_shape=tf.reshape(label_shape_raw,[2])

    map_raw = tf.decode_raw(map_raw, tf.uint8)
    map_shape_raw = tf.decode_raw(map_shape_raw, tf.int32)
    map_shape=tf.reshape(map_shape_raw,[2])

    image=tf.reshape(image_raw,img_shape)
    label=tf.reshape(label_raw,label_shape)
    map=tf.reshape(map_raw,map_shape)

    return image,label,map


write_tfr_sal("E:\Archive\datasets\image process\STEREO-1000\STEREO-1000\\RGB","E:\Archive\datasets\image process\STEREO-1000\STEREO-1000\\GT",'E:\Archive\datasets\image process\STEREO-1000\STEREO-1000\\tfr','train')