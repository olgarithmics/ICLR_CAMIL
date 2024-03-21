from __future__ import absolute_import
from __future__ import print_function
import h5py
import os
from tensorflow.keras.preprocessing.image import Iterator
import numpy as np
import itertools
from flushed_print import print
import multiprocessing

class ImgIterator(Iterator):
    def __init__(self,
                 h5_files,
                 batch_size,
                 directory=".",
                 shuffle=False,
                 seed=None,
                 data_format='channels_last',
                 color_mode='rgb',
                 target_size=(256,256)):

        self.color_mode = color_mode
        self.directory = directory
        self.h5_files=h5_files
        self.current_filename_index=0
        self.data_format = data_format
        self.target_size = target_size
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        def _recursive_list(subpath):
            fname = []
            for root, _, files in os.walk(subpath):
                for file in files:
                    if file.lower().endswith('.' + "h5"):
                        fname.append(os.path.join(root, file))
            return fname

        def open_h5_file(file_path):

            with h5py.File(file_path, "r") as hdf5_file:

            #with h5py.File(os.path.join(directory, file_path), "r") as hdf5_file:
                samples = hdf5_file['coords'].shape[0]

            return samples, file_path, samples * [os.path.join(directory, file_path), ]

        #self.h5_files = _recursive_list(self.directory)
        pool = multiprocessing.pool.ThreadPool()

        self.samples=[]
        self.files=[]


        self.samples, self.files, filenames = zip(*pool.map(open_h5_file, self.h5_files))


        self.sample_indices = dict(zip(self.files, self.samples))
        filenames  = [ values*[os.path.join(directory, key), ]for key, values in self.sample_indices.items()]
        self.filenames = list(itertools.chain(*filenames))

        print('Found %d h5 files.' % (len(self.sample_indices.keys())))
        print('Found %d instances.' % (len(self.filenames)))

        pool.close()
        pool.join()
        super(ImgIterator, self).__init__(len(self.filenames), batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0
        self.current_filename_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()
                self.current_filename_index = 0

            if self.n == 0:
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n

            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
                batch_remainder = self.n - (self.batch_size * ((self.n + self.batch_size - 1) // self.batch_size))
                self.current_filename_index = self.n + batch_remainder

            self.total_batches_seen += 1

            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def _set_index_array(self):

        indices = [np.arange(values) for key, values in self.sample_indices.items()]
        self.index_array = np.concatenate(indices, axis=0)


    def _get_batches_of_transformed_samples(self, index_array):


        fnames = self.filenames[self.current_filename_index:self.current_filename_index + index_array.shape[0]]

        unique_filenames = sorted(set(fnames), key=fnames.index)

        self.current_filename_index += index_array.shape[0]


        if len(unique_filenames) > 1:

            bag_indices = np.split(index_array, np.where(index_array == 0)[0])

            bag_features = []

            for bag_index, fname in zip(bag_indices, unique_filenames):
                if bag_index.size == 0:
                    continue
                with h5py.File(fname, "r") as hdf5_file:

                    bag_features.append(hdf5_file['imgs'][bag_index[0]:bag_index[0] + bag_index.shape[0]])
            batch_x = np.concatenate(bag_features, axis=0)
        else:
            with h5py.File(unique_filenames[0], "r") as hdf5_file:

                batch_x = hdf5_file['imgs'][index_array[0]:index_array[0] + index_array.shape[0]]
        batch_x = np.array(batch_x, dtype="float32")
        #batch_x=preprocess_input(batch_x)
        batch_x /= 255
        return batch_x

def load_images(iterator,num_child=4, workers=8):

    while True:
        for batch_images in iterator:
            for i in range(num_child):
                yield batch_images

def discriminator_loader(img_loader, latent_dim=256, seed=0):
    rng = np.random.RandomState(seed)
    while True:
        x = next(img_loader)
        batch_size = x.shape[0]
        # Sample z from isotropic Gaussian
        z_p = rng.normal(size=(batch_size, latent_dim))

        y_real = np.ones((batch_size,), dtype='float32')
        y_fake = np.zeros((batch_size,), dtype='float32')

        yield [x, z_p], [y_real, y_fake, y_fake]

def decoder_loader(img_loader, latent_dim=256, seed=0):
    rng = np.random.RandomState(seed)
    while True:
        x = next(img_loader)
        batch_size = x.shape[0]
        # Sample z from isotropic Gaussian
        z_p = rng.normal(size=(batch_size, latent_dim))
        # Label as real
        y_real = np.ones((batch_size,), dtype='float32')
        yield [x, z_p], [y_real, y_real]

def encoder_loader(img_loader):
    while True:
        x = next(img_loader)
        yield x, None



