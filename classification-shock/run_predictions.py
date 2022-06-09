import csv
from pickle import load
import numpy as np
import pandas as pd
from tensorflow import keras
from datetime import datetime
from speasy import amda, config
from utils import get_features_from_amda

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

FEATURES_TO_USE = [(TOTELS_1_INDEX, TOTELS_1_ID), (TOTELS_8_INDEX, TOTELS_8_ID), (TOTELS_6_INDEX, TOTELS_6_ID),
                   (RHO_ID_INDEX, RHO_ID)]

TOLERANCE_START_STOP_TIMES = 10
WINDOW_WIDTH_AROUND_SHOCK = 90  # minutes
THRESHOLD_PROBABILITY_CLASSIFICATION = 0.90

# Tolerance to merge features based on pandas dataframe' indexes
TOLERANCE_FEATURES_CONCATENATING = 4  # 4 seconds
BATCH_SIZE = 2048

# For TEST
YEAR_TEST = 2008
MONTH_1_TEST = 7
MONTH_2_TEST = 7
DAY_1_TEST = 3
DAY_2_TEST = 5

START_TIME_TEST = datetime(YEAR_TEST, MONTH_1_TEST, DAY_1_TEST)
STOP_TIME_TEST = datetime(YEAR_TEST, MONTH_2_TEST, DAY_2_TEST)

if __name__ == '__main__':
    # Login to amda
    config.amda_username.set('testKernel')
    config.amda_password.set('amda')

    amda_list_user_parameters = amda.list_user_parameters()
    print(amda_list_user_parameters)

    model = keras.models.load_model('model.h5')
    # Check its architecture
    model.summary()
    # load the scaler
    scaler = load(open('scaler.pkl', 'rb'))
    # Get dataset for test
    data_set_for_testing = get_features_from_amda(amda_list_user_parameters, FEATURES_TO_USE, START_TIME_TEST,
                                                  STOP_TIME_TEST, tolerance=TOLERANCE_FEATURES_CONCATENATING)

    data_set_for_testing = data_set_for_testing.dropna()
    list_index_test_bis = data_set_for_testing.index.tolist()

    test_features_bis = np.array(data_set_for_testing.copy())

    test_features_bis = scaler.transform(test_features_bis)

    test_predictions_baseline_bis = model.predict(test_features_bis, batch_size=BATCH_SIZE)
    test_labels_bis = np.zeros(len(test_predictions_baseline_bis))

    feature1_id_to_show = FEATURES_TO_USE[0][1]

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

    events_file_name = "Bow_Shock_Events_" + str(YEAR_TEST) + "_" + str(
        THRESHOLD_PROBABILITY_CLASSIFICATION) + "_more_trained "
    with open(events_file_name + '.csv', 'w') as f:
        write = csv.writer(f, delimiter=" ")
        write.writerows(list_events)
