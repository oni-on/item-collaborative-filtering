"""
Item to Item Collaborative Filtering
Paper: https://www.computer.org/csdl/mags/ic/2017/03/mic2017030012.pdf
"""
from itertools import permutations

import numpy as np


class ItemItemCollaborativeFiltering:

    def __init__(self, item_column='item_id', user_column='user_id', transaction_column='transaction_id'):
        self.item_column = item_column
        self.user_column = user_column
        self.transaction_column = transaction_column

    def df_to_dict(self, df, key, value, return_frozen_set=True):
        """
        Transforms a pandas dataframe

        df = pd.DataFrame({
        'key':['A', 'A', 'B', 'C', 'B'],
        'value': [1, 2, 3, 3, 5]
        })

        with columns [key, value] to a dictionary with the following structure

        where,

        dict key = key column in dataframe
        dict value =

        {
        A: frozenset(1, 2,),
        B: frozenset(3, 5),
        C: frozenset(3)
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

        return dict(zip(unique_keys, values_arrays))

    def __generate_item_pairs(self, df, item):
        """
        Creates permutations/pairs of two products
        :param df: dataframe with columns [user_id, item_id, transaction_id]
        :param item: item that is paired with all other items, if None - all possible pairs are created
        :return: list of tuples, length=df['item'].nunique()
        """
        if item is not None:
            return [(item, paired_item) for paired_item in df[self.item_column].unique() if paired_item!=item]
        else:
            return list(permutations(df[self.item_column].unique(), 2))


    def __count_common_item_pair_users(self, df, item1, item2):
        """
        Computes the number of users that interacted (e.g. watched, purchased) BOTH with item1 and item2
        :param df: dataframe with columns [user_id, item_id, transaction_id]
        :param item1: item id
        :param item2: item id
        :return: int, number of users in common for the item pair (item1, item2)
        """
        item1_users = set(df.loc[df[self.item_column]==item1, self.user_column].unique())
        item2_users = set(df.loc[df[self.item_column]==item2, self.user_column].unique())

        common_users = len(item1_users.intersection(item2_users))

        return common_users
