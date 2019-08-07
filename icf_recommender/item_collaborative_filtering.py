"""
Item Based Collaborative Filtering
"""

# Author Oni On <oni.on.qepa@gmail.com>

from itertools import permutations

import numpy as np
import pandas as pd


class ItemCollaborativeFiltering:
    """
    Item Based Collaborative Filtering: a recommendation algorithm by Amazon

    In a nutshell, the algorithm looks as follows:

    1. Create pairs of items (X, Y) using all the items in the catalog.
    2. For every item pair (X, Y) compute a score for the statement
    "Users who viewed/watched/bought X, also viewed/watched/bought Y"
    3. For a given item X, the items Y are sorted according to the score in a decreasing order.
    The top items Y are recommended to all users who viewed/watched/bought product X.

    For more details check the project `Wiki <https://github.com/oni-on/item-collaborative-filtering/wiki>`

    Parameters
    ----------
    item_column: str, default='item_id'. The name of the item id column in the dataframe
    user_column: str, default='user_id'. The name of the user id column in the dataframe

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import icf_recommender
    >>> df = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv")
    >>> df['liked'] = np.where(df.rating >= 5, 1, 0)
    >>> df_liked = df.loc[df.liked == 1, ["user_id", "book_id"]]
    >>> recommender = icf_recommender.ItemCollaborativeFiltering(item_column='book_id')
    >>> df_recommendations = recommender.fit_recommendations(df=df_liked, item=4081)
    >>> recommended_books = recommender.recommend(df_recommendations, 4081)

    Attributes
    ----------

    References
    ----------
    https://www.computer.org/csdl/mags/ic/2017/03/mic2017030012.pdf
    """

    def __init__(self, item_column='item_id', user_column='user_id'):
        """
        :param item_column: str, default='item_id'. The name of the item id column in the input dataframe
        :param user_column: str, default='user_id'. The name of the user id column in the input dataframe
        """
        self.item_column = item_column
        self.user_column = user_column
        self.df_recommendations = pd.DataFrame()

    def __generate_item_pairs(self, df, item):
        """
        Creates permutations/pairs of two products
        :param df: dataframe with columns [user_id, item_id]
        :param item: item that is paired with all other items, if None - all possible pairs are created
        :return: list of tuples, length=df['item'].nunique()
        """
        if item is not None:
            return [(item, paired_item) for paired_item in df[self.item_column].unique() if paired_item != item]
        else:
            return [(item, paired_item) for
                    item, paired_item in permutations(df[self.item_column].unique(), 2)]

    def __calculate_item_users(self, df):
        """
        Creates a set of the users that interacted (e.g. watched, purchased) with an item
        Sets self.item_users to dictionary {item1: set(user1, user2, ...), item2: set()...}
        :param df: dataframe with columns [user_id, item_id]
        """

        item_users = df.groupby(
            self.item_column
        ).agg(
            {
                self.user_column: lambda x: set(x)
            }
        ).to_dict()[self.user_column]

        self.item_users = item_users

    def __count_common_item_pair_users(self, item_pair):
        """
        Computes the number of users that interacted (e.g. watched, purchased) with BOTH items in a pair
        :param item_pair: tuple(item1, item2)
        :return: (tuple, int) = (item_pair, number of users in common for the item pair)
        """
        item1, item2 = item_pair
        try:
            item1_users = self.item_users[item1]
            item2_users = self.item_users[item2]
            common_users_count = len(item1_users.intersection(item2_users))
            return item_pair, common_users_count
        except AttributeError:
            "Extract item users first, using __item_users()"

    def __calculate_item_probabilities(self, df):
        """
        Computes the probability of a user interacting with an item (e.g. watching, purchasing, liking)
        Sets self.item_probabilities to dictionary with key=item_id, value = interaction probability (float)
        :param df:  dataframe with columns [user_id, item_id]
        """
        item_probabilities = (df.groupby(
            self.item_column
        ).agg(
            {
                self.user_column: 'nunique'
            }
        )/df[self.user_column].nunique()).to_dict()[self.user_column]

        self.item_probabilities = item_probabilities

    def __item_interaction_probability(self, item):
        """
        Looks up the probability of a user interacting with an item (e.g. watching, purchasing, liking)
        :param item: item_id
        :return: float, probability
        """
        try:
            return self.item_probabilities[item]
        except AttributeError:
            "Extract item probabilities first, using __item_probabilities()"

    def __calculate_user_interactions(self, df):
        """
        For ALL users computes the number of items that every user interacted with, APART from item in the df row
        Sets self.user_interactions to a dictionary with key = user_id, value = number of interactions
        :param df: dataframe with columns [user_id, item_id]
        """

        # subtract 1 to count number of interactions with other items DIFFERENT from item
        interactions_count = df.groupby(
            self.user_column
        )[self.item_column].agg('nunique') - 1

        self.user_interactions = interactions_count.to_dict()

    def __count_users_interactions(self, df, item):
        """
        For ALL users of an item, it looks up the number of other items that every user interacted with, APART from item
        :param df: dataframe with columns [user_id, item_id]
        :param item: the item_id to be looked up
        :return: array of integers with number of interactions per user. length = nr of users that interacted with item.
        """

        try:
            filtered_users = df.loc[df[self.item_column] == item, self.user_column].values
            interactions_count = np.array([self.user_interactions[user] for user in filtered_users])
            return interactions_count
        except AttributeError:
            "Extract user interactions first, using __user_interactions()"

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

        item1, item2 = item_pair

        product_probability = self.__item_interaction_probability(item2)

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

    def fit_recommendations(self, df, item=None):
        """
        Computes recommendation strength for item pairs
        By default item=None, means recommendations are computed for all items
        :param df: dataframe with columns [user_id, item_id]
        :param item: item that is scored with all other items, if None - all possible item pairs are scored
        :return: dataframe with columns [item, recommended_item, actual_common_users, expected_common_users, score]
        """
        item_pairs = self.__generate_item_pairs(df, item)

        self.__calculate_item_users(df)
        self.__calculate_item_probabilities(df)
        self.__calculate_user_interactions(df)

        # output: [((item1, item2), common_users)]
        count_pair_users = [self.__count_common_item_pair_users(item_pair) for item_pair in item_pairs]

        # filter out item pairs with no users in common
        count_pair_users = list(
           filter(
               lambda x: x[1] > 0,
               count_pair_users
           )
        )

        # extract item pair, and user count
        filtered_item_pairs, count_pair_users = zip(*count_pair_users)

        # output: [expected_users]
        # compute expected users for item pairs with at least 1 user in common
        expected_pair_users = [self.__expected_common_item_pair_users(df, item_pair)
                               for item_pair in filtered_item_pairs]

        # recommendation score function
        pair_score = self.__recommendations_score_function(np.array(expected_pair_users), np.array(count_pair_users))

        items, recommended_items = zip(*filtered_item_pairs)

        df_recommendations = pd.DataFrame({
            'item': items,
            'recommended_item': recommended_items,
            'count_common_users': count_pair_users,
            'expected_common_users': expected_pair_users,
            'score': pair_score
        })

        return df_recommendations

    @staticmethod
    def recommend(df_recommendations, item, n_recommendations=10):
        """
        Returns item recommendations
        :param df_recommendations: dataframe with recommendations generated with fit_recommendations()
        :param item: the item for which recommendations are generated
        :param n_recommendations: the number of generated recommendations
        :return: list of recommended products
        """

        recommended_items = df_recommendations[df_recommendations.item == item]
        recommended_items = recommended_items.sort_values(
            'score', ascending=False
        ).groupby(
            'item'
        ).head(n_recommendations)['recommended_item'].values
        return recommended_items


