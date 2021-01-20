import pickle

from workflows import Workflow

with open('data/real_datasets.pkl', 'rb') as fp:
    datasets = pickle.load(fp)

delta = 0.002

results = dict()
for df in datasets:
    
    y = data.target.values
    X = data.drop(['target'], axis=1)
    
    small_data_streams = ['AbruptInsects',
                          'Insects',
                          'Keystroke',
                          'ozone',
                          'outdoor',
                          'luxembourg']

    if str(df) in small_data_streams:
        n_train_obs = 500
        W = n_train_obs
    else:
        n_train_obs = 1000
        W = n_train_obs
    
    predictions, detections, train_size, training_info, results_comp = \
        Workflow(X=X, y=y,delta=delta,window_size=W)
    
    ds_results = \
        dict(predictions=predictions,
             detections=detections,
             n_updates=train_size,
             data_size=len(y),
             training_info=training_info,
             results_comp=results_comp)
    
    results[df] = ds_results
    
    with open('data/studd_experiments.pkl', 'wb') as fp:
        pickle.dump(results, fp)
