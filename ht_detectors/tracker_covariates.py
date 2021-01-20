import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class FeatTracker(object):

    def __init__(self, ref_window, thr, window_size):
        self.method = ks_2samp
        self.alarm_list = []
        self.data = []
        self.ref_window = np.array(ref_window)
        self.window_size = window_size
        self.thr = thr
        self.index = 0
        self.p_value = 1

    def add_element(self, elem):
        self.data.append(elem)

    def detected_change(self):
        if len(self.data) < self.window_size:
            self.index += 1
            return False

        x = np.array(self.data)
        x = x[-self.window_size:]
        w = self.ref_window

        ht = self.method(x, w)
        p_value = ht[1]
        has_change = p_value < self.thr
        self.p_value = p_value

        if has_change:
            # print('Change detected at index: ' + str(self.index))
            self.alarm_list.append(self.index)
            self.index += 1
            return True
        else:
            self.index += 1
            return False


class XCTracker(object):

    def __init__(self, X, thr, W=None):
        """
        X change tracker
        :param X: pd df
        """

        self.X = X
        self.col_names = list(self.X.columns)
        self.trackers = dict.fromkeys(self.col_names)
        self.thr = thr
        self.index = 0
        self.p_values = None
        if W is None:
            self.W = self.X.shape[0]
        else:
            self.W = W

        self.X = self.X.tail(self.W)

    def create_trackers(self):
        for col in self.trackers:
            x = np.array(self.X.loc[:, col])

            self.trackers[col] = \
                FeatTracker(ref_window=x,
                            thr=self.thr,
                            window_size=self.W)

    def reset_trackers(self):
        self.trackers = dict.fromkeys(self.col_names)
        self.X = self.X.tail(self.W)
        self.create_trackers()

    def get_p_values(self):

        self.p_values = \
            [self.trackers[x].p_value
             for x in self.trackers]

    def add_element(self, Xi):
        Xi_df = pd.DataFrame(Xi)
        Xi_df.columns = self.X.columns
        self.X.append(Xi_df, ignore_index=True)

        x = Xi.flatten()

        for i, col in enumerate(self.col_names):
            self.trackers[col].add_element(x[i])

    def detected_change(self):

        self.index += 1

        changes = []
        for col in self.col_names:
            has_change = \
                self.trackers[col].detected_change()

            changes.append(has_change)

        changes = np.array(changes)

        any_change = np.any(changes)

        if any_change:
            print('Change detected at index: ' + str(self.index))
            self.reset_trackers()

        return any_change
