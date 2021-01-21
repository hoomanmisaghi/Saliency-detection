# Saliency-detection
Saliency detection in pixel level is a segementation task of finding most important objects in the image. Different CNNs for saliency detection are in this respitory. The code in this respitory has been completed in the time span of my masters, so it is not a production code and all parts might not be consistant. 
different approaches to tackle the saliency detection task has been implemented here all based on convolutional neural networks, the main goal of the project was making a CNN that is accurate and at the same time light weight that can be used on medium processors of commercial robots. The implemented approaches are as follows:
Saliency detection in pixel level using CNN with dilated convolutions
it is a Tensorflow class of a convolutional neural network for saliency detection task in pixel level.
The designed network has fully convolutional architecture which consists of a encoder and decoder part. This network uses Pyramid pooling as described in Deeplab 3.
One of the main features of provided class in this project is having many features from simple network creation to traing and network optimization for deployment which can be used as the structure in tensorflow 1 projects. 
