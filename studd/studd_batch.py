from skmultiflow.data.data_stream import DataStream
from skmultiflow.drift_detection.page_hinkley import PageHinkley as PHT
from ht_detectors.tracker_output import HypothesisTestDetector
import copy
import numpy as np


class STUDD:

    def __init__(self, X, y, n_train):
        """

        :param X:
        :param y:
        :param n_train:
        """

        D = DataStream(X, y)
        D.prepare_for_use()

        self.datastream = D
        self.n_train = n_train
        self.W = n_train
        self.base_model = None
        self.student_model = None
        self.init_training_data = None

    def initial_fit(self, model, std_model):
        """

        :return:
        """

        X_tr, y_tr = self.datastream.next_sample(self.n_train)

        model.fit(X_tr, y_tr)

        yhat_tr = model.predict(X_tr)

        std_model.fit(X_tr, yhat_tr)

        self.base_model = model
        self.student_model = std_model
        self.init_training_data = dict({"X": X_tr, "y": y_tr, "y_hat": yhat_tr})

    DETECTOR = PHT

    @staticmethod
    def drift_detection_std(datastream_, model_,
                            std_model_, n_train_,
                            delta, n_samples,
                            upd_model=False,
                            upd_std_model=True,
                            detector=DETECTOR):

        datastream = copy.deepcopy(datastream_)
        base_model = copy.deepcopy(model_)
        student_model = copy.deepcopy(std_model_)
        n_train = copy.deepcopy(n_train_)

        std_detector = detector(delta=delta)
        std_alarms = []

        iter = n_train
        n_updates = 0
        samples_used = 0
        y_hat_hist = []
        y_buffer, y_hist = [], []
        X_buffer, X_hist = [], []
        while datastream.has_more_samples():
            # print("Iteration: " + str(iter))

            Xi, yi = datastream.next_sample()
            y_hist.append(yi[0])
            y_buffer.append(yi[0])
            X_hist.append(Xi[0])
            X_buffer.append(Xi[0])

            model_yhat = base_model.predict(Xi)
            y_hat_hist.append(model_yhat[0])
            std_model_yhat = student_model.predict(Xi)

            std_err = int(model_yhat != std_model_yhat)
            std_detector.add_element(std_err)

            if std_detector.detected_change():
                print("Found change std in iter: " + str(iter))
                std_alarms.append(iter)

                if upd_model:
                    X_buffer = np.array(X_buffer)
                    y_buffer = np.array(y_buffer)

                    samples_used_iter = len(y_buffer[-n_samples:])

                    print("Updating model with " + str(samples_used_iter), " Observations")
                    base_model.fit(X_buffer[-n_samples:],
                                   y_buffer[-n_samples:])

                    yhat_buffer = base_model.predict(X_buffer)

                    if upd_std_model:
                        student_model.fit(X_buffer, yhat_buffer)
                    else:
                        student_model.fit(X_buffer[-n_samples:],
                                          yhat_buffer[-n_samples:])

                    # y_buffer = []
                    # X_buffer = []
                    y_buffer = list(y_buffer)
                    X_buffer = list(X_buffer)

                    n_updates += 1
                    samples_used += samples_used_iter
                    print("Moving on")

            iter += 1

        preds = dict({"y": y_hist, "y_hat": y_hat_hist})

        output = dict({"alarms": std_alarms,
                       "preds": preds,
                       "n_updates": n_updates,
                       "samples_used": samples_used})

        return output

    @staticmethod
    def drift_detection_spv(datastream_, model_, n_train_,
                            delay_time, observation_ratio,
                            delta, n_samples,
                            upd_model=False,
                            detector=DETECTOR):
        import copy
        import numpy as np

        datastream = copy.deepcopy(datastream_)
        model = copy.deepcopy(model_)
        n_train = copy.deepcopy(n_train_)

        driftmodel = detector(delta=delta)
        alarms = []

        iter = n_train
        j, n_updates, samples_used = 0, 0, 0
        yhat_hist = []
        y_buffer, y_hist = [], []
        X_buffer, X_hist = [], []
        while datastream.has_more_samples():
            # print("Iteration: " + str(iter))

            Xi, yi = datastream.next_sample()

            y_hist.append(yi[0])
            y_buffer.append(yi[0])
            X_hist.append(Xi[0])
            X_buffer.append(Xi[0])

            model_yhat = model.predict(Xi)

            yhat_hist.append(model_yhat[0])

            put_i_available = np.random.binomial(1, observation_ratio)

            if put_i_available > 0:
                if j >= delay_time:
                    err = int(y_hist[j - delay_time] != yhat_hist[j - delay_time])
                    driftmodel.add_element(err)

            if driftmodel.detected_change():
                print("Found change in iter: " + str(iter))
                alarms.append(iter)

                if upd_model:
                    X_buffer = np.array(X_buffer)
                    y_buffer = np.array(y_buffer)

                    samples_used_iter = len(y_buffer[-n_samples:])

                    print("Updating model with " + str(samples_used_iter), " Observations")
                    model.fit(X_buffer[-n_samples:],
                              y_buffer[-n_samples:])

                    y_buffer = list(y_buffer)
                    X_buffer = list(X_buffer)

                    n_updates += 1
                    samples_used += samples_used_iter
                    print("Moving on")

            iter += 1
            j += 1

        preds = dict({"y": y_hist, "y_hat": yhat_hist})

        output = dict({"alarms": alarms,
                       "preds": preds,
                       "n_updates": n_updates,
                       "samples_used": samples_used})

        return output

    @staticmethod
    def BL2_retrain_after_w(datastream_, model_, n_train_, n_samples):
        import copy
        import numpy as np

        datastream = copy.deepcopy(datastream_)
        model = copy.deepcopy(model_)
        n_train = copy.deepcopy(n_train_)
        iter = copy.deepcopy(n_train_)

        j, n_updates, samples_used = 0, 0, 0
        yhat_hist = []
        y_buffer, y_hist = [], []
        X_buffer, X_hist = [], []
        while datastream.has_more_samples():
            # print("Iteration: " + str(iter))

            Xi, yi = datastream.next_sample()

            y_hist.append(yi[0])
            y_buffer.append(yi[0])
            X_hist.append(Xi[0])
            X_buffer.append(Xi[0])

            model_yhat = model.predict(Xi)

            yhat_hist.append(model_yhat[0])

            if iter % n_train == 0 and iter > n_train + 1:
                X_buffer = np.array(X_buffer)
                y_buffer = np.array(y_buffer)

                samples_used_iter = len(y_buffer[-n_samples:])

                print("Updating model with " + str(samples_used_iter), " Observations")
                model.fit(X_buffer[-n_samples:],
                          y_buffer[-n_samples:])

                y_buffer = list(y_buffer)
                X_buffer = list(X_buffer)

                n_updates += 1
                samples_used += samples_used_iter
                print("Moving on")

            iter += 1
            j += 1

        preds = dict({"y": y_hist, "y_hat": yhat_hist})

        output = dict({"alarms": [],
                       "preds": preds,
                       "n_updates": n_updates,
                       "samples_used": samples_used})

        return output

    @staticmethod
    def BL1_never_adapt(datastream_, model_):
        import copy

        datastream = copy.deepcopy(datastream_)
        model = copy.deepcopy(model_)

        yhat_hist, y_hist = [], []

        while datastream.has_more_samples():
            # print("Iteration: " + str(iter))

            Xi, yi = datastream.next_sample()

            y_hist.append(yi[0])

            model_yhat = model.predict(Xi)

            yhat_hist.append(model_yhat[0])

        preds = dict({"y": y_hist, "y_hat": yhat_hist})

        output = dict({"alarms": [],
                       "preds": preds,
                       "n_updates": 0,
                       "samples_used": 0})

        return output

    @staticmethod
    def drift_detection_uspv(datastream_, model_, n_train_,
                             use_prob,
                             method,
                             pvalue,
                             window_size,
                             n_samples,
                             upd_model=False):
        import copy
        import numpy as np

        assert method in ["wrs", "tt", "ks"]

        datastream = copy.deepcopy(datastream_)
        model = copy.deepcopy(model_)
        n_train = copy.deepcopy(n_train_)

        driftmodel = HypothesisTestDetector(method=method,
                                            window=window_size,
                                            thr=pvalue)
        alarms = []

        y_buffer = []
        y_hist = []
        X_buffer = []
        y_hat_hist = []
        n_updates = 0
        samples_used = 0

        iter = n_train
        while datastream.has_more_samples():
            # print("Iteration: " + str(iter))

            Xi, yi = datastream.next_sample()

            y_buffer.append(yi[0])
            y_hist.append(yi[0])
            X_buffer.append(Xi[0])

            y_hat_hist.append(model.predict(Xi)[0])

            if use_prob:

                yprob_all = model.predict_proba(Xi)

                if len(yprob_all) < 2:
                    yhat = yprob_all[0]
                elif len(yprob_all) == 2:
                    yhat = yprob_all[1]
                else:
                    yhat = np.max(yprob_all)
            else:
                yhat = model.predict(Xi)[0]

            driftmodel.add_element(yhat)

            if driftmodel.detected_change():
                print("Found change in iter: " + str(iter))
                alarms.append(iter)

                if upd_model:
                    X_buffer = np.array(X_buffer)
                    y_buffer = np.array(y_buffer)

                    samples_used_iter = len(y_buffer[-n_samples:])

                    print("Updating model with " + str(samples_used_iter), " Observations")
                    model.fit(X_buffer[-n_samples:],
                              y_buffer[-n_samples:])

                    # y_buffer = []
                    # X_buffer = []
                    y_buffer = list(y_buffer)
                    X_buffer = list(X_buffer)

                    n_updates += 1
                    samples_used += samples_used_iter
                    print("Moving on")

            iter += 1

        preds = dict({"y": y_hist, "y_hat": y_hat_hist})

        output = dict({"alarms": alarms,
                       "preds": preds,
                       "n_updates": n_updates,
                       "samples_used": samples_used})

        return output

    

    @staticmethod
    def drift_detection_uspv_f(datastream_, model_, n_train_,
                               use_prob,
                               method,
                               pvalue,
                               window_size,
                               n_samples,
                               upd_model=False):
        import copy
        import numpy as np

        from ht_detectors.tracker_output import FixedWindowDetector

        assert method in ["wrs", "tt", "ks"]

        datastream = copy.deepcopy(datastream_)
        model = copy.deepcopy(model_)
        n_train = copy.deepcopy(n_train_)

        driftmodel = FixedWindowDetector(ref_window=[], thr=pvalue, window_size=window_size)
        alarms = []

        y_buffer = []
        y_hist = []
        X_buffer = []
        y_hat_hist = []
        n_updates = 0
        samples_used = 0

        iter = n_train
        while datastream.has_more_samples():
            # print("Iteration: " + str(iter))

            Xi, yi = datastream.next_sample()

            y_buffer.append(yi[0])
            y_hist.append(yi[0])
            X_buffer.append(Xi[0])

            y_hat_hist.append(model.predict(Xi)[0])

            if use_prob:

                yprob_all = model.predict_proba(Xi)

                if len(yprob_all) < 2:
                    yhat = yprob_all[0]
                elif len(yprob_all) == 2:
                    yhat = yprob_all[1]
                else:
                    yhat = np.max(yprob_all)
            else:
                yhat = model.predict(Xi)[0]

            driftmodel.add_element(yhat)

            if driftmodel.detected_change():
                print("Found change in iter: " + str(iter))
                alarms.append(iter)

                if upd_model:
                    X_buffer = np.array(X_buffer)
                    y_buffer = np.array(y_buffer)

                    samples_used_iter = len(y_buffer[-n_samples:])

                    print("Updating model with " + str(samples_used_iter), " Observations")
                    model.fit(X_buffer[-n_samples:],
                              y_buffer[-n_samples:])

                    # y_buffer = []
                    # X_buffer = []
                    y_buffer = list(y_buffer)
                    X_buffer = list(X_buffer)

                    n_updates += 1
                    samples_used += samples_used_iter
                    print("Moving on")

            iter += 1

        preds = dict({"y": y_hist, "y_hat": y_hat_hist})

        output = dict({"alarms": alarms,
                       "preds": preds,
                       "n_updates": n_updates,
                       "samples_used": samples_used})

        return output

    @staticmethod
    def drift_detection_uspv_x(datastream_, model_, n_train_,
                               X,
                               pvalue,
                               window_size,
                               n_samples,
                               upd_model=False):
        import copy
        import numpy as np

        from ht_detectors.tracker_covariates import XCTracker

        datastream = copy.deepcopy(datastream_)
        model = copy.deepcopy(model_)
        n_train = copy.deepcopy(n_train_)

        driftmodel = XCTracker(X=X, thr=pvalue, W=window_size)
        driftmodel.create_trackers()

        alarms = []

        y_buffer = []
        y_hist = []
        X_buffer = []
        y_hat_hist = []
        n_updates = 0
        samples_used = 0

        iter = n_train
        while datastream.has_more_samples():
            # print("Iteration: " + str(iter))

            Xi, yi = datastream.next_sample()

            y_buffer.append(yi[0])
            y_hist.append(yi[0])
            X_buffer.append(Xi[0])

            y_hat_hist.append(model.predict(Xi)[0])

            # yhat = model.predict(Xi)[0]

            driftmodel.add_element(Xi)

            if driftmodel.detected_change():
                print("Found change in iter: " + str(iter))
                alarms.append(iter)

                if upd_model:
                    X_buffer = np.array(X_buffer)
                    y_buffer = np.array(y_buffer)

                    samples_used_iter = len(y_buffer[-n_samples:])

                    print("Updating model with " + str(samples_used_iter), " Observations")
                    model.fit(X_buffer[-n_samples:],
                              y_buffer[-n_samples:])

                    # y_buffer = []
                    # X_buffer = []
                    y_buffer = list(y_buffer)
                    X_buffer = list(X_buffer)

                    n_updates += 1
                    samples_used += samples_used_iter
                    print("Moving on")

            iter += 1

        preds = dict({"y": y_hist, "y_hat": y_hat_hist})

        output = dict({"alarms": alarms,
                       "preds": preds,
                       "n_updates": n_updates,
                       "samples_used": samples_used})

        return output
