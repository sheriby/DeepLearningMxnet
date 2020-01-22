from mxnet.gluon import data as gdata
import sys
import time
import d2lzh as d2l

mnist_train = gdata.vision.FashionMNIST(train=True) 
# testing data set just for evaluate the model rather than training
mnist_test = gdata.vision.FashionMNIST(train=False)

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
                   'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# draw images and its label
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # _ means that we will not use this varible
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy()) # draw a image
        f.set_title(lbl) # text label
        f.axes.get_xaxis().set_visible(False) # display x-axis
        f.axes.get_yaxis().set_visible(False) # display y-axis

def load_data_fashion_mnist(batch_size):        
    transformer = gdata.vision.transforms.ToTensor() 
    # ToTensor=>transform image date format from 'unit8' to 'float32' and divided by 255 ensure each data in 0-1 
    if sys.platform.startswith('win'): # it is windows operator system
        num_workers = 0 # 0 means that not use extra process to accelerate reading data
    else:
        num_workers = 4 # linux, yeah! That's good!

    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                batch_size, shuffle=True,
                                num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                batch_size, shuffle=True,
                                num_workers=num_workers)
    return train_iter, test_iter