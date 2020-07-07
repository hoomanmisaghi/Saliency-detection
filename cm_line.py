from net_class import AtrousNet
import os
from PIL import Image
import time
net=AtrousNet()

# net.train(20,20)

print(net.create())


# for test



# # #
image_ref_folder_random='E:\Archive\image process\saliency setection\MSRA-B\img'


img_folder='E:\Archive\image process\saliency setection\MSRA-B\img'
out_path='E:\python learning\my inception like\\atrous_2\\image output'

# HKU
# image_ref_folder_random='E:\Archive\image process\HKU-IS\data\imgs'
#
#
# img_folder='E:\Archive\image process\HKU-IS\data\imgs'
# out_path='E:\python learning\my inception like\\atrous_3\image output hku'
#

img_list=os.listdir(image_ref_folder_random)


# #
# for i,img_name in enumerate(img_list):
#     print(i)
#
#     name=img_name[:-4]
#     img = os.path.join(img_folder, name+'.jpg')
#     p=net.use(img)
#
#     p=Image.fromarray(p*255)
#     p=p.convert(mode='L')
#     #p.show()
#     p.save(os.path.join(out_path,name+'.png'),'PNG')
#
#


#
times=[]
net.input_node='Placeholder:0'
net.predict_node='sal_net/Conv_11/Sigmoid'
net.freez_pre_trained()
net.predict_node='sal_net/Conv_11/Sigmoid:0'
net.load_graph()
net.sess_from_graph()
for i,img_name in enumerate(img_list):
    print(i)

    name=img_name[:-4]
    img = os.path.join(img_folder, name+'.jpg')
    p,t=net.use_pb(img)
    times.append(t)
    p=Image.fromarray(p*255)
    p=p.convert(mode='L')
    # p.show()
    p.save(os.path.join(out_path,name+'.png'),'PNG')

import numpy as np

print(1/np.mean(times))
