"""API to handle data and associated labels."""

import csv

import data_constants as dc
import numpy as np
import pandas as pd

#debugging
from matplotlib import pyplot as plt

class Data():

    def __init__(self, data_path, labels_path):
        self.current_index = 0
        self.data = self.load_data_from_csv(data_path, labels_path)

    def _load_image(self, path):
        """Converts a png image to a pixel array, normalizes from 0 to 1.

        For two frames.

        Args:
            path: str, location of the frames in the form base_name1_name2
        Returns:
            np pixel array
        """

        pb = gtk.gdk.pixbuf_new_from_file(path)
        y = dc.GATHER_HEIGHT - dc.HEIGHT
        x = dc.WIDTH
        pb_cropped = pb.subpixbuf(x, y, dc.WIDTH, dc.HEIGHT)
        pix = pb_cropped.get_pixels_array().flatten()

        # Debugging
        # print pb_cropped.get_height()
        # print pb_cropped.get_width()
        # print len(pix)
        # pix = pix.reshape(dc.HEIGHT, dc.WIDTH, 3)
        # plt.imshow(pix, interpolation='nearest')
        # plt.show()

        return pix/dc.MAX_PIX_VAL

    def _load_csv(self, path):
        """Loads the csv map of images and labels. Get names of maps from csv.

        Args:
            path: str, location of the csv map.
        Returns:
            {filename : label}
        """
        csv_df = pd.read_csv(path)
        d = {'filepath': csv_df['base'] + csv_df['name'].map(str) + '.png',
             'label': csv_df['damage']}
        return pd.DataFrame(d)

    def _load_onehot(self, index):
        """Returns a one hot numpy array encoded from index.

        Args:
            index: which num to make one hot
        """
        onehot = np.zeros(dc.ONEHOT_SIZE, dtype=np.int)
        onehot.itemset(index, 1)
        return onehot

    def get_next_batch(self, size):
        """Returns the next batch of size of image, label pairs.

        Loops around if needed.

        Args:
            size: int, size of next batch
        """
        prev_index = self.current_index
        self.current_index += size
        if self.current_index >= len(self.data):
            self.current_index -= len(self.data)
            return pd.concat([self.data.iloc[prev_index:],
                              self.data.iloc[:self.current_index]])
        else:
            return self.data.iloc[prev_index:self.current_index]

    def load_data_from_csv(self, data_path, labels_path):
        """Loads all data and labels from csv map.

        Args:
            path: str, location of the csv map.
        Returns:
            [([pixel array], label)]
        """

        print "Opening CSV from %s" % path
        csv_df = self._load_csv(path)

        d = {'data': csv_df['filepath'].map(self._load_image),
             'label': csv_df['label'].map(self._load_onehot)}

        data_df = pd.DataFrame(d)

        print "Data size: %d" % len(data_df)

        return data_df
