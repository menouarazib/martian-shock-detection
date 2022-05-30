import csv
from datetime import timedelta, datetime
from pickle import dump

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from speasy import amda, config
from speasy.inventory.data_tree import amda as amdatree
from matplotlib import pyplot as plt, colors
import seaborn as sns

from tensorflow import keras
from utils import catalog_to_dataframe, get_features_from_amda, get_list_interval_shocks, get_dataset_concatenated, \
    get_labels, make_model, plot_metrics, plot_cm

# Parameters of features
TOTELS_1_INDEX = 4
TOTELS_1_ID = "ws_totels_1"

TOTELS_8_ID = "ws_totels_8"
TOTELS_8_INDEX = 6

TOTELS_6_ID = "ws_totels_6"
TOTELS_6_INDEX = 5

MEX_MARS_ID = "mex_mars_r"

RHO_ID = "ws_rho"
RHO_ID_INDEX = 7

FEATURES = [(TOTELS_1_INDEX, TOTELS_1_ID), (TOTELS_8_INDEX, TOTELS_8_ID), (None, MEX_MARS_ID), (RHO_ID_INDEX, RHO_ID)]

FEATURES_TO_USE = [(TOTELS_1_INDEX, TOTELS_1_ID), (TOTELS_8_INDEX, TOTELS_8_ID), (TOTELS_6_INDEX, TOTELS_6_ID),
                   (RHO_ID_INDEX, RHO_ID)]

# Tolerance to merge features based on pandas dataframe' indexes
TOLERANCE_FEATURES_CONCATENATING = 4  # 4 seconds

# For Train
YEAR = 2012
MONTH_1 = 7
MONTH_2 = 7
DAY_1 = 1
DAY_2 = 15

START_TIME_TRAIN = datetime(YEAR, MONTH_1, DAY_1)
STOP_TIME_TRAIN = datetime(YEAR, MONTH_2, DAY_2)

# For TEST
YEAR_TEST = 2011
MONTH_1_TEST = 7
MONTH_2_TEST = 7
DAY_1_TEST = 3
DAY_2_TEST = 5

START_TIME_TEST = datetime(YEAR_TEST, MONTH_1_TEST, DAY_1_TEST)
STOP_TIME_TEST = datetime(YEAR_TEST, MONTH_2_TEST, DAY_2_TEST)
# Parameters of getting and adjusting data
TOLERANCE_START_STOP_TIMES = 10
WINDOW_WIDTH_AROUND_SHOCK = 90  # minutes

# Column's name of the shock in dataframe
EVENT_LABEL = "event"

# Useful constants
YEAR_LABEL = 'year'
MONTH_LABEL = 'month'
DAY_LABEL = 'day'
DATA_SHOCK_LABEL = 'data'

THRESHOLD_PROBABILITY_CLASSIFICATION = 0.90  # 0.75  # 0.82

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]

colors_list = list(colors._colors_full_map.values())

EPOCHS = 300
BATCH_SIZE = 2048

if __name__ == '__main__':
    # Login to amda
    config.amda_username.set('testKernel')
    config.amda_password.set('amda')

    amda_list_user_parameters = amda.list_user_parameters()
    print(amda_list_user_parameters)

    # Get from AMDA mex shocks events
    mex_shocks = list(amda.get_catalog(amdatree.Catalogs.SharedCatalogs.MARS.MEXShockCrossings))
    mex_shocks.sort(key=lambda ev: ev.start_time.timestamp())
    mex_shocks_df = catalog_to_dataframe(mex_shocks)
    mex_shocks_df[YEAR_LABEL] = pd.DatetimeIndex(mex_shocks_df[DATA_SHOCK_LABEL]).year
    mex_shocks_df = mex_shocks_df[mex_shocks_df[YEAR_LABEL] == YEAR]
    mex_shocks_df[MONTH_LABEL] = pd.DatetimeIndex(mex_shocks_df[DATA_SHOCK_LABEL]).month
    mex_shocks_df = mex_shocks_df[(mex_shocks_df[MONTH_LABEL] >= MONTH_1) & (mex_shocks_df[MONTH_LABEL] <= MONTH_2)]
    if MONTH_1 == MONTH_2:
        mex_shocks_df[DAY_LABEL] = pd.DatetimeIndex(mex_shocks_df[DATA_SHOCK_LABEL]).day
        mex_shocks_df = mex_shocks_df[(mex_shocks_df[DAY_LABEL] >= DAY_1) & (mex_shocks_df[DAY_LABEL] <= DAY_2)]

    print("Nb shocks: ", len(mex_shocks_df))

    start_time_features = mex_shocks_df[DATA_SHOCK_LABEL].iloc[0] - timedelta(
        minutes=WINDOW_WIDTH_AROUND_SHOCK + TOLERANCE_START_STOP_TIMES)
    end_time_features = mex_shocks_df[DATA_SHOCK_LABEL].iloc[-1] + timedelta(
        minutes=WINDOW_WIDTH_AROUND_SHOCK + TOLERANCE_START_STOP_TIMES)

    # Get dataset based on features
    data_set_for_training = get_features_from_amda(amda_list_user_parameters, FEATURES_TO_USE, start_time_features,
                                                   end_time_features, tolerance=TOLERANCE_FEATURES_CONCATENATING)

    union_DatetimeIndex = get_list_interval_shocks(mex_shocks_df, WINDOW_WIDTH_AROUND_SHOCK)

    start_time_features = start_time_features.replace(tzinfo=None)
    end_time_features = end_time_features.replace(tzinfo=None)

    data_set_for_training = get_dataset_concatenated(data_set_for_training, union_DatetimeIndex)

    del union_DatetimeIndex

    # Get labels
    labels = get_labels(data_set_for_training, mex_shocks_df, nb_neighbors=0)
    data_set_for_training[EVENT_LABEL] = np.array(labels)
    data_set_for_training[EVENT_LABEL] = data_set_for_training[EVENT_LABEL].astype(np.int8)

    # Delete NaN rows
    data_set_for_training = data_set_for_training.dropna()

    data_set_for_training.to_pickle("dataset+_"+str(DAY_1)+"_"+str(DAY_2)+".pkl")
    exit(0)

    neg, pos = np.bincount(data_set_for_training[EVENT_LABEL])
    total = neg + pos
    print('Class imbalanced:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    # print(data_set_for_training)

    # Use a utility from sklearn to split our dataset.
    train_df, test_df = train_test_split(data_set_for_training, test_size=0.2, shuffle=False)
    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=False)

    # Form np arrays of labels and features.
    train_labels = np.array(train_df.pop(EVENT_LABEL))
    bool_train_labels = train_labels != 0
    val_labels = np.array(val_df.pop(EVENT_LABEL))
    test_labels = np.array(test_df.pop(EVENT_LABEL))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)

    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    print('Training labels shape:', train_labels.shape)
    print('Validation labels shape:', val_labels.shape)
    print('Test labels shape:', test_labels.shape)

    print('Training features shape:', train_features.shape)
    print('Validation features shape:', val_features.shape)
    print('Test features shape:', test_features.shape)

    pos_df = pd.DataFrame(train_features[bool_train_labels], columns=train_df.columns)
    neg_df = pd.DataFrame(train_features[~bool_train_labels], columns=train_df.columns)

    feature1_id_to_show = FEATURES_TO_USE[0][1]
    feature2_id_to_show = FEATURES_TO_USE[-1][1]

    sns.jointplot(x=pos_df[feature1_id_to_show], y=pos_df[feature2_id_to_show],
                  kind='hex', xlim=(-5, 5), ylim=(-5, 5))
    plt.suptitle("Positive distribution")

    sns.jointplot(x=neg_df[feature1_id_to_show], y=neg_df[feature2_id_to_show],
                  kind='hex', xlim=(-5, 5), ylim=(-5, 5))
    _ = plt.suptitle("Negative distribution")
    plt.show()

    model = make_model(train_features_shape=train_features.shape[-1], metrics=METRICS)
    print(model.summary())

    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    baseline_history = model.fit(
        train_features,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_features, val_labels), class_weight=class_weight)

    model.save('my_model_more_trained.h5')
    dump(scaler, open('scaler_more_trained.pkl', 'wb'))
    exit(0)
    plot_metrics(baseline_history, colors_list)
    plt.show()

    test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)

    baseline_results = model.evaluate(test_features, test_labels,
                                      batch_size=BATCH_SIZE, verbose=0)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    plot_cm(test_labels, test_predictions_baseline, p=THRESHOLD_PROBABILITY_CLASSIFICATION)
    plt.show()

    non_shocks, shocks = np.bincount(test_labels)
    """
    For TESTING
    """
    # Get dataset for test
    data_set_for_testing = get_features_from_amda(amda_list_user_parameters, FEATURES_TO_USE, START_TIME_TEST,
                                                  STOP_TIME_TEST, tolerance=TOLERANCE_FEATURES_CONCATENATING)

    data_set_for_testing = data_set_for_testing.dropna()
    list_index_test_bis = data_set_for_testing.index.tolist()

    test_features_bis = np.array(data_set_for_testing.copy())

    test_features_bis = scaler.transform(test_features_bis)

    test_predictions_baseline_bis = model.predict(test_features_bis, batch_size=BATCH_SIZE)
    test_labels_bis = np.zeros(len(test_predictions_baseline_bis))

    plot_cm(test_labels_bis, test_predictions_baseline_bis, p=THRESHOLD_PROBABILITY_CLASSIFICATION)
    plt.show()

    list_ipv_test_filtered = []
    for i_loop in range(len(list_index_test_bis)):
        p = test_predictions_baseline_bis[i_loop]
        value_i = data_set_for_testing[feature1_id_to_show][i_loop]
        if p >= THRESHOLD_PROBABILITY_CLASSIFICATION and value_i > 0:
            index_i = list_index_test_bis[i_loop]
            probability_i = p
            list_ipv_test_filtered.append((index_i, probability_i, value_i))

    i = 0
    j = 1
    size_window_around = 40  # minutes
    list_df = []
    current_list = []
    while i < len(list_ipv_test_filtered):
        stop = False
        index_time_i, _, _ = list_ipv_test_filtered[i]
        current_list.append(list_ipv_test_filtered[i])
        if j == len(list_ipv_test_filtered):
            list_df.append(current_list)
            break
        while not stop and j < len(list_ipv_test_filtered):
            index_time_j, _, _ = list_ipv_test_filtered[j]
            print(index_time_i, index_time_j)
            if index_time_j <= index_time_i + pd.DateOffset(minutes=size_window_around):
                current_list.append(list_ipv_test_filtered[j])
            else:
                stop = True
                list_df.append(current_list)
                current_list = []
                i = j
            j = j + 1

    list_dfs = []
    column_probability = 'Probability'
    column_value = 'Value'
    for list_loop in list_df:
        list_indexes = []
        list_probabilities = []
        list_values = []
        for i_loop in range(len(list_loop)):
            index_i, probability, value = list_loop[i_loop]
            list_indexes.append(index_i)
            list_probabilities.append(probability)
            list_values.append(value)

        df = pd.DataFrame(index=list_indexes, data=np.column_stack([list_probabilities, list_values]),
                          columns=[column_probability, column_value])
        list_dfs.append(df)

    list_events = []
    previous_date = None
    for df_i in list_dfs:
        column_p = df_i[column_probability]
        column_v = df_i[column_value]
        max_p = column_p.max()
        if max_p < THRESHOLD_PROBABILITY_CLASSIFICATION:
            continue

        max_index = column_p.idxmax()
        max_value = column_v[max_index]
        start_time_shock = max_index
        if previous_date is not None and start_time_shock <= previous_date + pd.DateOffset(minutes=size_window_around):
            previous_date = start_time_shock
            continue
        previous_date = start_time_shock

        end_time_shock = start_time_shock + pd.DateOffset(minutes=1)
        start_time_shock = str(start_time_shock.isoformat())
        end_time_shock = str(end_time_shock.isoformat())

        list_events.append(
            [start_time_shock[:19], end_time_shock[:19], str(max_p),
             str(max_value)])

    events_file_name = "train_" + str(DAY_1) + "_" + str(DAY_2) + "test_" + str(DAY_1_TEST) + "_" + str(
        DAY_2_TEST) + "_" + str(MONTH_2_TEST) + "_" + str(
        YEAR_TEST) + "_probThreshold_" + str(
        THRESHOLD_PROBABILITY_CLASSIFICATION) + "_timeThreshold_" + str(size_window_around)
    events_file_name = "Bow_Shock_Events_"+str(YEAR_TEST)+str(THRESHOLD_PROBABILITY_CLASSIFICATION)
    with open(events_file_name + '.csv', 'w') as f:
        write = csv.writer(f, delimiter=" ")
        write.writerows(list_events)
