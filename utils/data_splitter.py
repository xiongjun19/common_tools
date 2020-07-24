# coding=utf8


from sklearn.model_selection import StratifiedShuffleSplit


def strat_field_split(X_arr, y_arr, rand_state, test_size=0.1):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=rand_state)
    res = sss.split(x_arr, y_arr)
    train_x, train_y = None, None
    test_x, test_y = None, None
    for train_idx, test_idx in res:
        train_x = x_arr[train_idx]
        train_y = y_arr[train_idx]
        test_x = x_arr[test_idx]
        test_y = y_arr[test_idx]
    return train_x, train_y, test_x, test_y

