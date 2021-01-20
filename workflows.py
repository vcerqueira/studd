import pandas as pd
import numpy as np
from studd.studd_batch import STUDD
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as RF

from skmultiflow.drift_detection.page_hinkley import PageHinkley as PHT



def Workflow(X, y, delta, window_size):
    ucdd = STUDD(X=X, y=y, n_train=window_size)

    ucdd.initial_fit(model=RF(), std_model=RF())

    print("Detecting change by tracking features")

    UFD = ucdd.drift_detection_uspv_x(datastream_=ucdd.datastream,
                                      model_=ucdd.base_model,
                                      n_train_=ucdd.n_train,
                                      X=X,
                                      window_size=window_size,
                                      pvalue=delta,
                                      n_samples=window_size,
                                      upd_model=True)

    print("Detecting change with STUDD")
    RES_STUDD = ucdd.drift_detection_std(datastream_=ucdd.datastream,
                                        model_=ucdd.base_model,
                                        std_model_=ucdd.student_model,
                                        n_train_=ucdd.n_train,
                                        n_samples=window_size,
                                        delta=delta / 2,
                                        upd_model=True,
                                        upd_std_model=True,
                                        detector=PHT)


    print("Detecting change with bl1")
    res_bl1 = ucdd.BL1_never_adapt(datastream_=ucdd.datastream,
                                   model_=ucdd.base_model)

    print("Detecting change with bl2")
    res_bl2 = ucdd.BL2_retrain_after_w(datastream_=ucdd.datastream,
                                       model_=ucdd.base_model,
                                       n_train_=ucdd.n_train,
                                       n_samples=window_size)

    print("Detecting change with SS")
    SS = ucdd.drift_detection_spv(datastream_=ucdd.datastream,
                                  model_=ucdd.base_model,
                                  n_train_=ucdd.n_train,
                                  n_samples=window_size,
                                  delay_time=0,
                                  observation_ratio=1,
                                  upd_model=True,
                                  delta=delta,
                                  detector=PHT)

    print("Detecting change with UTH")
    UHT = ucdd.drift_detection_uspv(datastream_=ucdd.datastream,
                                    model_=ucdd.base_model,
                                    n_train_=ucdd.n_train,
                                    use_prob=False,
                                    n_samples=window_size,
                                    method="ks",
                                    window_size=window_size,
                                    upd_model=True,
                                    pvalue=delta)

    print("Detecting change with UTHF")
    UHTF = ucdd.drift_detection_uspv_f(datastream_=ucdd.datastream,
                                       model_=ucdd.base_model,
                                       n_train_=ucdd.n_train,
                                       use_prob=False,
                                       n_samples=window_size,
                                       method="ks",
                                       window_size=window_size,
                                       upd_model=True,
                                       pvalue=delta)

    DSS = ucdd.drift_detection_spv(datastream_=ucdd.datastream,
                                   model_=ucdd.base_model,
                                   n_train_=ucdd.n_train,
                                   delay_time=int(window_size / 2),
                                   n_samples=window_size,
                                   observation_ratio=1,
                                   upd_model=True,
                                   delta=delta,
                                   detector=PHT)

    WS = ucdd.drift_detection_spv(datastream_=ucdd.datastream,
                                  model_=ucdd.base_model,
                                  n_train_=ucdd.n_train,
                                  n_samples=window_size,
                                  delay_time=0,
                                  observation_ratio=.5,
                                  upd_model=True,
                                  delta=delta,
                                  detector=PHT)

    DWS = ucdd.drift_detection_spv(datastream_=ucdd.datastream,
                                   model_=ucdd.base_model,
                                   n_train_=ucdd.n_train,
                                   n_samples=window_size,
                                   delay_time=int(window_size / 2),
                                   observation_ratio=0.5,
                                   upd_model=True,
                                   delta=delta,
                                   detector=PHT)

    training_info = ucdd.init_training_data

    results = {
               "STUDD": RES_STUDD,
               "BL1": res_bl1,
               "BL2": res_bl2,
               "SS": SS,
               "DSS": DSS,
               "WS": WS,
               "DWS": DWS,
               "UHT": UHT,
               "UHTF": UHTF,
               "UFD": UFD}

    perf_kpp = dict()
    perf_acc = dict()
    nupdates = dict()
    pointsbought = dict()
    for m in results:
        x = results[m]
        perf_acc_i = metrics.accuracy_score(y_true=x["preds"]["y"],
                                            y_pred=x["preds"]["y_hat"])

        perf_m = metrics.cohen_kappa_score(y1=x["preds"]["y"],
                                           y2=x["preds"]["y_hat"])

        pointsbought[m] = x["samples_used"]
        nupdates[m] = x["n_updates"]

        perf_kpp[m] = perf_m
        perf_acc[m] = perf_acc_i

    perf_kpp = pd.DataFrame(perf_kpp.items())
    perf_acc = pd.DataFrame(perf_acc.items())

    perf = pd.concat([perf_kpp.reset_index(drop=True), perf_acc], axis=1)

    perf.columns = ['Method', 'Kappa', 'rm', 'Acc']
    perf = perf.drop("rm", axis=1)

    return perf, pointsbought, nupdates, training_info, results

