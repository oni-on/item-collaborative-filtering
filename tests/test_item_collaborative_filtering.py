import unittest

import numpy as np
import pandas as pd

from src.item_collaborative_filtering import ItemItemCollaborativeFiltering

class TestItemItemCollaborativeFiltering(unittest.TestCase):

    def setUp(self):
        self.item_item_collaborative_filtering = ItemItemCollaborativeFiltering()

    def test_fit_recommendations(self):
        df = pd.DataFrame({'item_id': ['A', 'A', 'A', 'A', 'B', 'B', 'B'],
                           'user_id': [1, 2, 3, 4, 1, 2, 5]})

        recommendations = self.item_item_collaborative_filtering.fit_recommendations(df)

        self.assertTrue(
            (recommendations.item.values==np.array(['A', 'B'])).all(),
            msg="Recommendations dataframe 'item' column values not as expected"
        )
        self.assertTrue(
            (recommendations.recommended_item.values==np.array(['B', 'A'])).all(),
            msg="Recommendations dataframe 'recommended_item' column values not as expected"
        )
        self.assertTrue(
            (recommendations.count_common_users.values==np.array([2, 2])).all(),
            msg="__count_common_item_pair_users() - unexpected results"
        )
        self.assertTrue(
            (recommendations.expected_common_users.values==np.array([6/5, 8/5])).all(),
            msg="__expected_common_item_pair_users() - unexpected results"
        )
        self.assertTrue(
            (recommendations.score.values.round(4)==np.array([0.5418, 0.2346])).all(),
            msg="recommendations scores - unexpected results"
        )

    def tearDown(self):
        pass


