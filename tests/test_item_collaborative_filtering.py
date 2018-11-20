import unittest

import pandas as pd

from src.item_collaborative_filtering import ItemItemCollaborativeFiltering

class TestItemItemCollaborativeFiltering(unittest.TestCase):

    def setUp(self):
        self.item_item_collaborative_filtering = ItemItemCollaborativeFiltering()

    def test_df_to_dict(self):
        df = pd.DataFrame({'key': ['A', 'A', 'B', 'C', 'B'],
                           'value': [1, 2, 3, 3, 5]})

        self.assertEqual(
            self.item_item_collaborative_filtering.df_to_dict(df, 'key', 'value'),
            {'A': frozenset([1, 2]), 'B': frozenset([3, 5]), 'C': frozenset([3])}
        )

    def tearDown(self):
        pass
