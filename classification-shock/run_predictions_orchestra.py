import csv
import sys
from pickle import load

import numpy as np
import pandas as pd
from tensorflow import keras

from utils import validate_time_format, get_features_from_amda
from speasy import amda, config

DATE_FORMAT = '%Y-%m-%d'

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

# Tolerance to merge features based on pandas dataframe' indexes
TOLERANCE_FEATURES_CONCATENATING = 4  # 4 seconds

BATCH_SIZE = 2048

THRESHOLD_PROBABILITY_CLASSIFICATION = 0.5

if __name__ == '__main__':
    arguments = sys.argv
    print("Given arguments: ", arguments)
    nb_arguments = len(arguments)
    destination_folder_path = arguments[1]
    start = arguments[2]
    stop = arguments[3]

    start = validate_time_format(start)
    stop = validate_time_format(stop)

    print("Given start date: ", start)
    print("Given end date: ", stop)

    # Login to amda
    config.amda_username.set('testKernel')
    config.amda_password.set('amda')

    amda_list_user_parameters = amda.list_user_parameters()

    model = keras.models.load_model('model.h5')
    # Check its architecture
    model.summary()
    # Load the scaler
    scaler = load(open('scaler.pkl', 'rb'))

    # Get dataset for test
    data_set_for_prediction = get_features_from_amda(amda_list_user_parameters, FEATURES_TO_USE, start, stop,
                                                     tolerance=TOLERANCE_FEATURES_CONCATENATING)

    data_set_for_prediction = data_set_for_prediction.dropna()

    list_indexes = data_set_for_prediction.index.tolist()
    features = np.array(data_set_for_prediction.copy())
    scaled_features = scaler.transform(features)

    predictions_baseline = model.predict(features, batch_size=BATCH_SIZE)
    labels = np.zeros(len(predictions_baseline))

    events_index = []
    for i_loop in range(len(list_indexes)):
        p = predictions_baseline[i_loop]
        value_i = data_set_for_prediction[FEATURES_TO_USE[0][1]][i_loop]
        if p >= THRESHOLD_PROBABILITY_CLASSIFICATION and value_i > 0:
            index_i = list_indexes[i_loop]
            probability_i = p
            events_index.append((index_i, probability_i, value_i))

    size_window_around = 40  # minutes
    _i = 0
    _j = 1
    list_events = []
    list_df = []
    current_list = []
    while _i < len(events_index):
        time, p, v = events_index[_i]
        if _i == len(events_index) - 1 or _j == len(events_index):
            list_df.append(current_list)
            break
        else:
            current_list.append(events_index[_i])
        while _j < len(events_index):
            time_j, p_j, v_j = events_index[_j]
            if time_j <= time + pd.DateOffset(minutes=size_window_around):
                current_list.append(events_index[_j])
                _j += 1
            else:
                _i = _j
                _j += 1
                list_df.append(current_list)
                current_list = []
                break

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

    for df_i in list_dfs:
        column_p = df_i[column_probability]
        column_v = df_i[column_value]

        max_index = column_v.idxmax()

        start_time_shock = max_index
        end_time_shock = start_time_shock + pd.DateOffset(minutes=1)
        start_time_shock = str(start_time_shock.isoformat())
        end_time_shock = str(end_time_shock.isoformat())

        list_events.append([start_time_shock[:19], end_time_shock[:19]])

    events_file_name = "Bow_Shock_Events_" + str(start.month) + "_" + str(start.year)

    # open the file in the write mode
    f = open(events_file_name + '.csv', 'w')
    # create the csv writer
    writer = csv.writer(f, delimiter=" ")

    for i in range(len(list_events)):
        row = list_events[i]
        # write a row to the csv file
        writer.writerow(row)

    # close the file
    f.close()
