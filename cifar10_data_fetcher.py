import tensorflow as tf
from math import ceil
from glob import glob
import pickle
import tarfile
import random
import os
from params import Cifar10Params

def Cifar10ConvertToTFRecords(image_params:Cifar10Params):
    """
    this function should be run only once, to convert the Cifar10 tar.gz file from https://www.cs.toronto.edu/~kriz/cifar.html to tfrecord files
    """
    data_dir = image_params.path
    tarfile.open(os.path.join(data_dir, 'cifar-10-python.tar.gz'),'r:gz').extractall(data_dir)
    file_names = {}
    file_names['train'] = ['data_batch_{:d}'.format(i) for i in range(1, 5)]
    file_names['validation'] = ['data_batch_5']
    file_names['test'] = ['test_batch']
    input_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
        output_file = os.path.join(data_dir, mode + '.tfrecords')
        try:
            os.remove(output_file)
        except OSError:
            pass
        with tf.python_io.TFRecordWriter(output_file) as record_writer:
            for input_file in input_files:
                with tf.gfile.Open(input_file, 'rb') as f:
                    data_dict = pickle.load(f,encoding='latin1')
                data = data_dict['data']
                labels = data_dict['labels']
                num_entries_in_batch = len(labels)
                for i in range(num_entries_in_batch):
                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data[i].tobytes()])),
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]]))
                        }))
                    record_writer.write(example.SerializeToString())

class Cifar10DataFetcher():
    """
    a wrapper for a tf.data.Dataset object that can feed Cifar10 images.
    An instance will have the properties `image` and `labels` which are the tensors for a batch of (normalized) images and one-hot labels.
    """
    def __init__(self, mode: str, params:Cifar10Params=Cifar10Params(), batch_size: int = None, noise_batch_size:int = None,order='NCHW'):
        """

        Args:
            mode: can be either `'TRAIN'`,`'VALIDATION'`,`'TEST'`. controls which data set to use.
            params: a `Cifar10Params` object containing parameters of the images
            batch_size: number of images per batch
            noise_batch_size: if `None`, the batch will be a four dimensional tensor, whose first dimension is the batch dimension with size `batch_size`. otherwise, the batch will be a five dimensional ternsor, whose first dimension is the noise batch dimension, with size `noise_batch_size`, and the second dimension is the image batch dimension, with size `batch_size`.
            order: either `'NCHW'` or `'NHWC'`. (currently only `'NHWC'` is supported for image augmentation preprocessing)
        """

        # choose correct path for images, between training, validation and test
        if mode.upper() in ['TRAIN','TRAINING']:
            mode = 'TRAIN'
            path = os.path.join(params.path,'train.tfrecords')
            is_training = True
        elif mode.upper() in ['VALIDATE', 'VALIDATION', 'VALIDATING']:
            mode = 'VALIDATION'
            path = os.path.join(params.path, 'validation.tfrecords')
            is_training = False
        elif mode.upper() in ['EVALUATE', 'EVAL', 'EVALUATION', 'EVALUATING', 'TEST', 'TESTING']:
            mode = 'TEST'
            path = os.path.join(params.path, 'test.tfrecords')
            is_training = False
        else:
            raise ValueError('wrong value for `mode`')

        self._graph = tf.get_default_graph()

        # get list of filenames of tfrecords
        if os.path.isdir(path):
            file_names = glob(path + "/**/*", recursive=True)
            if is_training:
                random.shuffle(file_names)
        else:
            file_names = [path]

        with self._graph.as_default():
            dataset = tf.data.TFRecordDataset(file_names)

            dataset = dataset.repeat()

            def preprocess_image(image, is_training):
                ##TODO: make this work for NCHW as well
                if is_training:
                    image = tf.image.resize_image_with_crop_or_pad(image, params.image_size + 8, params.image_size+ 8)
                    image = tf.random_crop(image, [params.image_size, params.image_size, params.number_of_channels])
                    image = tf.image.random_flip_left_right(image)
                return tf.image.per_image_standardization(image)

            def parse_record(raw_record, is_training):
                features = tf.parse_single_example(raw_record,features={
                        'image': tf.FixedLenFeature([], tf.string),
                        'label': tf.FixedLenFeature([], tf.int64),
                    })
                image = tf.decode_raw(features['image'], tf.uint8)
                image = tf.reshape(image, [params.number_of_channels,params.image_size,params.image_size])
                image = tf.transpose(image, [1, 2, 0]) # to NHWC
                image = tf.cast(image,tf.float32)
                image = preprocess_image(image, is_training)
                if order=='NCHW':
                    image = tf.transpose(image, [2, 0, 1]) # to NCHW
                label = tf.one_hot(tf.cast(features['label'], tf.int32),params.num_classes)
                return image, label

            dataset = dataset.map(lambda value: parse_record(value, is_training))

            if (batch_size is None):
                batch_size = 1

            if is_training:
                # TODO: I just took this formula from somewhere, it should be empirical justified...
                dataset = dataset.shuffle(int(params.training_set_size*0.4+3*batch_size))

            if noise_batch_size is None:
                total_batch_size = batch_size
            else:
                total_batch_size = batch_size*noise_batch_size

            if mode == 'VALIDATION':
                set_size = params.validation_set_size
            else:
                set_size = params.test_set_size
            if (set_size % total_batch_size) != 0:
                raise ValueError("in modes 'VALIDATION' and 'TEST', total batch size should divide set size")
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(total_batch_size))

            if noise_batch_size is not None:
                def reshape(images, labels):
                    if order == 'NHWC':
                        images = tf.reshape(images, [noise_batch_size, batch_size, params.image_size, params.image_size, params.number_of_channels])
                    if order == 'NCHW':
                        images = tf.reshape(images, [noise_batch_size, batch_size, params.number_of_channels, params.image_size, params.image_size])
                    labels = tf.reshape(labels, [noise_batch_size, batch_size, params.num_classes])
                    return images, labels
                dataset = dataset.map(lambda images,labels: reshape(images,labels))

            self._next = dataset.make_one_shot_iterator().get_next()
            self.image, self.label = self._next

    def GetNext(self, sess: tf.Session = None):
        """
        get next batch of images and labels
        Args:
            sess: a tf.Session

        Returns:
            images, labels

        """
        if sess is None:
            sess = tf.get_default_session()

        image, label = sess.run(self._next)
        return image, label