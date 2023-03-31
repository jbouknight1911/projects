"""
The main code for the Strings-to-Vectors assignment. See README.md for details.
"""
from typing import Sequence, Any

import numpy as np


class Index:
    """
    Represents a mapping from a vocabulary (e.g., strings) to integers.
    """

    def __init__(self, vocab: Sequence[Any], start=0):
        """
        Assigns an index to each unique item in the `vocab` iterable,
        with indexes starting from `start`.

        Indexes should be assigned in order, so that the first unique item in
        `vocab` has the index `start`, the second unique item has the index
        `start + 1`, etc.
        """
        self.vocab = vocab
        self.start = start
        self.unique_items = set()
        self.index = start
        self.result = {}

        for item in vocab:
            if item not in self.unique_items:
                self.unique_items.add(item)
                self.result[item] = self.index
                self.index += 1

        self.object_list = list(self.result.keys())
        self.index_list = list(self.result.values())

    def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a vector of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array of the object indexes.
        """

        indexes = np.empty(0)

        for obj in object_seq:

            indexes = np.append(indexes, int(self.result.get(obj, self.start - 1)))

        return indexes

    def objects_to_index_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a matrix of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        If the sequences are not all of the same length, shorter sequences will
        have padding added at the end, with `start-1` used as the pad value.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array of the object indexes.
        """
        bin_matrix = []
        max_length = -999999  # not sure if we could import sys and use maxunicode here instead

        for object_seq in object_seq_seq:
            if len(object_seq) > max_length:
                max_length = len(object_seq)

        for object_seq in object_seq_seq:
            row = []

            for item in object_seq:
                row.append(self.result.get(item, self.start - 1))

            while len(row) < max_length:
                row.append(self.start - 1)

            bin_matrix.append(row)

        return np.array(bin_matrix)

    def objects_to_binary_vector(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a binary vector, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """

        # initialize vector of zeroes with appropriate size

        binary_vector = np.zeros(len(self.vocab) + self.start)

        for item in object_seq:
            if item in self.result:
                binary_vector[self.result.get(item)] = 1

        return binary_vector

    def objects_to_binary_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a binary matrix, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array, where each row in the array corresponds
                 to a row in the input, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """
        bin_matrix = []

        for object_seq in object_seq_seq:
            # Handling the shift differently. np.zeros was giving me data type issues
            row = []
            if self.start > 0:
                i = self.start

                while i > 0:
                    row.append(0)
                    i -= 1

            for obj_key in self.result:
                if obj_key in object_seq:
                    row.append(1)
                else:
                    row.append(0)

            bin_matrix.append(row)

        return np.array(bin_matrix)

    def indexes_to_objects(self, index_vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of objects associated with the indexes in the input
        vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_vector: A 1-dimensional array of indexes
        :return: A sequence of objects, one for each index.
        """

        objects = []

        for i in index_vector:
            if i in self.index_list:
                objects.append(self.object_list[self.index_list.index(i)])
        return objects

    def index_matrix_to_objects(
            self, index_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects associated with the indexes
        in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_matrix: A 2-dimensional array of indexes
        :return: A sequence of sequences of objects, one for each index.
        """

        object_seq = []

        for item in index_matrix:
            row = []
            for index in item:
                if index in self.index_list:
                    row.append(self.object_list[self.index_list.index(index)])

            object_seq.append(row)

        return object_seq

    def binary_vector_to_objects(self, vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of the objects identified by the nonzero indexes in
        the input vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param vector: A 1-dimensional binary array
        :return: A sequence of objects, one for each nonzero index.
        """
        object_seq = []

        # using a separate index here rather than start since we
        # just want to identify 1's (not objects) in the binary vector.

        i = 0

        for value in vector:
            if value == 1:
                object_seq.append(self.object_list[self.index_list.index(i)])
            i += 1
        return object_seq

    def binary_matrix_to_objects(
            self, binary_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects identified by the nonzero
        indices in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param binary_matrix: A 2-dimensional binary array
        :return: A sequence of sequences of objects, one for each nonzero index.
        """

        object_seq = []

        for bin_seq in binary_matrix:
            row = []
            nonzero_tuple = np.nonzero(bin_seq)

            for item in nonzero_tuple:
                for i in item:
                    row.append(self.object_list[self.index_list.index(i)])
                object_seq.append(row)

        return object_seq
