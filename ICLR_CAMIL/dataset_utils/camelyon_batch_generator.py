import numpy as np
import tensorflow as tf
import os
import h5py
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, filenames, args, shuffle=False, train=True, batch_size=1):
        self.filenames = filenames

        self.train = train
        self.batch_size = batch_size
        self.k = args.k
        self.shuffle = shuffle
        self.label_file = args.label_file
        self.on_epoch_end()

        self.enc = OneHotEncoder(handle_unknown='ignore')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames)))

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.filenames))

        if self.train == True:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        "returns one element from the data_set"
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_temp = [self.filenames[k] for k in indices]

        X, f, y = self.__data_generation(list_IDs_temp)

        return (X, f), np.array(y, dtype=np.float32)

    def __data_generation(self, filenames):
        """
        Parameters
        ----------
        batch_train:  a list of lists, each of which contains an np.ndarray of the patches of each image,
        the label of each image and a list of filenames of the patches
        Returns
        -------
        bag_batch: a list of np.ndarrays of size (numnber of patches,h,w,d) , each of which contains the patches of an image
        neighbors: a list  of the adjacency matrices of size (numnber of patches,number of patches) of every image
        bag_label: an np.ndarray of size (number of patches,1) reffering to the label of the image
        """

        for i in range(len(filenames)):

            with h5py.File(filenames[i], "r") as hdf5_file:

                base_name = os.path.splitext(os.path.basename(filenames[i]))[0]

                features = hdf5_file['features'][:]

                #sparse_coords = hdf5_file['adj_coords'][:]
                neighbor_indices = hdf5_file['indices'][:]

                values = hdf5_file['similarities'][:]

                if self.shuffle:
                    randomize = np.arange(neighbor_indices.shape[0])
                    np.random.shuffle(randomize)
                    features = features[randomize]
                    neighbor_indices = neighbor_indices[randomize]
                    values = values[randomize]

                references = pd.read_csv(self.label_file)

                bag_label = references["slide_label"].loc[references["slide_id"] == base_name].values.tolist()[0]

        #sparse_matrix = self.get_affinity(neighbor_indices[:, :4])

        Idx = neighbor_indices[:, :self.k]
        rows = np.asarray([[enum] * len(item) for enum, item in enumerate(Idx)]).ravel()

        columns = Idx.ravel()

        values = np.reshape(values, (neighbor_indices.shape[0], neighbor_indices.shape[1]))

        neighbor_matrix = values[:, 1:]
        normalized_matrix = preprocessing.normalize(neighbor_matrix, norm="l2")

        similarities = np.exp(-normalized_matrix)

        values = np.concatenate((np.max(similarities, axis=1).reshape(-1, 1), similarities), axis=1)

        values = values[:, :self.k]

        values = values.ravel().tolist()

        sparse_coords= list(zip(rows, columns))

        sparse_matrix = tf.sparse.SparseTensor(indices=sparse_coords,
                                               values=values,
                                               dense_shape=[features.shape[0], features.shape[0]])
        sparse_matrix = tf.sparse.reorder(sparse_matrix)

        return features, sparse_matrix, bag_label


