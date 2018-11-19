"""
Item to Item Collaborative Filtering
"""
import numpy as np


class ItemItemCollaborativeFiltering:

    def __init__(self):
        return self

    def df_to_dict(self, df, key, value, return_frozen_set=True):
        """
        Transforms a pandas dataframe

        df = pd.DataFrame({
        'key':['A', 'A', 'B', 'C', 'B'],
        'value': [1, 2, 3, 3, 5]
        })

        with columns [key, value] to a dictionary with the following structure

        {
        A: set(1, 2,),
        B: set(3, 5),
        C: set(3)
        }

        OR (if return_frozen_set=False)

        {
        A: [1, 2],
        B: [3, 5],
        C: [3]
        }

        :param df: dataframe that will be converted to dictionary
        :param key: (string) dataframe column name that will be the dictionary key
        :param value: (string) dataframe column name that will be the dictionary values
        :param return_frozen_set: (boolean) determines if dictionary values data type will be frozenset or array

        :return: dict
        """

        # get arrays with keys and values
        keys, values = df[[key, value]].sort_values(key).values.T

        # return: tuple(array1, array2) where array1: unique keys, array2: indices of unique keys
        unique_keys, unique_keys_indices = np.unique(keys, True)

        # split array into multiple arrays, each containing the values corresponding to unique_keys_index
        values_arrays = np.split(values, unique_keys_indices[1:])

        # convert each array into a frozenset
        if return_frozen_set:
            values_arrays = [frozenset(a) for a in values_arrays]

        return values_arrays










