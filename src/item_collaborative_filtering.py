"""
Item to Item Collaborative Filtering
Paper: https://www.computer.org/csdl/mags/ic/2017/03/mic2017030012.pdf
"""
from itertools import permutations

from joblib import Parallel, delayed
import numpy as np
import pandas as pd


class ItemItemCollaborativeFiltering:

    def __init__(self, item_column='item_id', user_column='user_id'):
        self.item_column = item_column
        self.user_column = user_column

    def __generate_item_pairs(self, df, item):
        """
        Creates permutations/pairs of two products
        :param df: dataframe with columns [user_id, item_id]
        :param item: item that is paired with all other items, if None - all possible pairs are created
        :return: list of tuples, length=df['item'].nunique()
        """
        if item is not None:
            return [(item, paired_item) for paired_item in df[self.item_column].unique() if paired_item!=item]
        else:
            return [(item, paired_item) for
                    item, paired_item in permutations(df[self.item_column].unique(), 2)]


    def __count_common_item_pair_users(self, df, item_pair):
        """
        Computes the number of users that interacted (e.g. watched, purchased) with BOTH items in a pair
        :param df: dataframe with columns [user_id, item_id]
        :param item_pair: tuple(item1, item2)
        :return: int, number of users in common for the item pair (item1, item2)
        """
        item1, item2 = item_pair
        item1_users = set(df.loc[df[self.item_column]==item1, self.user_column].unique())
        item2_users = set(df.loc[df[self.item_column]==item2, self.user_column].unique())

        common_users_count = len(item1_users.intersection(item2_users))

        return item_pair, common_users_count

    def __item_interaction_probability(self, df, item):
        """
        Computes the probability of a user interacting with an item (e.g. watching, purchasing, liking)
        :param df: dataframe with columns [user_id, item_id]
        :param item:
        :return: float, probability
        """
        return df.loc[df[self.item_column]==item, self.user_column].nunique()/df[self.user_column].nunique()

    def __count_users_interactions(self, df, item):
        """
        For ALL users of an item, it computes the number of other items that every user interacted with, APART from item
        :param df: dataframe with columns [user_id, item_id]
        :param item:
        :return: array of integers = number of interactions. length = nr of users that interacted with item.
        """

        filtered_users = df.loc[df[self.item_column] == item, self.user_column]

        # subtract 1 to count number of interactions with other items DIFFERENT from item
        interactions_count = df[df[self.user_column].isin(filtered_users)].groupby(
            self.user_column,
            group_keys=False
        )[self.item_column].agg('nunique').values - 1

        return interactions_count

    def __expected_common_item_pair_users(self, df, item_pair):
        """
        item_pair consists of (item1, item2)

        For users of item 1, it computes the EXPECTED number of users who interact (e.g. watch, purchase) with BOTH
        item1 and item2

        Mathematical explanation (bear with me...):
        Keep in mind that:  When two events, A and B, are independent, the probability of both occurring is:
        P(A and B) = P(A)xP(B)

        Given the pair (item1, item2):
        - the probability of a user interacting with item2 is p2 = (Users of item2)/All Users
        - the probability of NOT interacting with item2 is (1-p2)
        - if interaction events are assumed to be independent, the probability of NOT interacting with item2 for ONE
        user of item1 is (1-p2)^interactions_count. This follows from (1-p2)x...x(1-p2) (interactions_count times)
        - the inverse of the above is [1 - (1-p2)^interactions_count] a.k.a the probability of ONE item1 user
        interacting with item2 in interactions_count interactions of this user
        - the expected value of the above expression for ALL users is sum[1 - (1-p2)^interactions_count], where the sum
        is taken across all users that interacted with item1

        :param df: dataframe with columns [user_id, item_id]
        :param item_pair: tuple(item1, item2)
        :return: float, expected number of users in common for the item pair (item1, item2)
        """

        item1, item2 = item_pair[0], item_pair[1]

        product_probability = self.__item_interaction_probability(df, item2)

        interactions_count = self.__count_users_interactions(df, item1)

        return np.sum(1 - (1 - product_probability) ** interactions_count)

    @staticmethod
    def __recommendations_score_function(expected_users, actual_users):
        """
        Given the actual and expected users of two items, it computes how related the items are
        :param expected_users: np.array
        :param actual_users: np.array
        :return:
        """
        return (actual_users - expected_users) * np.log(actual_users + 0.1) / np.sqrt(expected_users)

    def fit_recommendations(self, df, item=None, processes=2):
       """
       Computes recommendation strength for item pairs
       By default item=None, means recommendations are computed for all items
       :param df: dataframe with columns [user_id, item_id]
       :param item: item that is scored with all other items, if None - all possible item pairs are scored
       :param processes: (int) number of processes used for parallel computation
       :return: dataframe with columns [item, recommended_item, actual_common_users, expected_common_users, score]
       """
       item_pairs = self.__generate_item_pairs(df, item)

       # output: [((item1, item2), common_users)]
       count_pair_users = Parallel(n_jobs=processes)(
           delayed(self.__count_common_item_pair_users)(df, item_pair) for item_pair in item_pairs
       )
       # filter out item pairs with no users in common
       count_pair_users = list(
           filter(
           lambda x: x[1] > 0,
               count_pair_users
           )
       )

       # extract item pair, and user count
       filtered_item_pairs = [item_pair[0] for item_pair in count_pair_users]
       count_pair_users = [users[1] for users in count_pair_users]

       # output: [expected_users]
       # compute expected users for item pairs with at least 1 user in common
       expected_pair_users = Parallel(n_jobs=processes)(
           delayed(self.__expected_common_item_pair_users)(df, item_pair) for item_pair in filtered_item_pairs
       )
       # recommendation score function
       pair_score = self.__recommendations_score_function(np.array(expected_pair_users), np.array(count_pair_users))

       items, recommended_items = zip(*filtered_item_pairs)

       df = pd.DataFrame({
           'item': items,
           'recommended_item': recommended_items,
           'count_common_users': count_pair_users,
           'expected_common_users': expected_pair_users,
           'score': pair_score
       })

       return df
