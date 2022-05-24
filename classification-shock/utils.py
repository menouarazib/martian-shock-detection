import math
from datetime import timedelta

import numpy as np
import pandas as pd
import speasy
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from speasy import amda
from tensorflow import keras


def catalog_to_dataframe(catalog, data_column_name='data'):
    """
    Converts an AMDA catalog to a dataframe
    @param catalog: a given amda catalog
    @param data_column_name: name of created column
    @return: a created dataframe
    """
    data = []
    for event in catalog:
        e = event.start_time
        data.append(e)
    df = pd.DataFrame(data, columns=[data_column_name], index=data)
    return df


def get_features_from_amda(amda_list_user_parameters, features_index_id, start_time, end_time, tolerance=60):
    """
    Get features from AMDA based on the given list of parameters using speasy
    @param amda_list_user_parameters: list of user's parameters stored in AMDA
    @param features_index_id: features indexes stored in AMDA
    @param start_time: start time
    @param end_time: stop time
    @param tolerance: tolerance of merging features together
    @return:
    """
    final_dataset = None
    for i_loop in range(len(features_index_id)):
        feature_index, feature_id = features_index_id[i_loop]
        if feature_index is not None:
            feature_df = amda.get_data(amda_list_user_parameters[feature_index],
                                       start_time, end_time).to_dataframe(datetime_index=True)
        else:
            feature_df = speasy.get_data('amda/' + feature_id, start_time, end_time).to_dataframe(datetime_index=True)

        feature_df[feature_id] = feature_df[feature_id].astype(np.float16)
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        if final_dataset is None:
            final_dataset = feature_df.copy()
            del feature_df
        else:
            final_dataset = pd.merge_asof(final_dataset, feature_df, left_index=True, right_index=True,
                                          tolerance=pd.Timedelta(str(tolerance) + 's'),
                                          allow_exact_matches=False)
            del feature_df
    return final_dataset


def get_list_interval_shocks(df_shocks, window_width):
    times_range_shocks = list()
    previous_date_range = None
    for i_shock, date_of_shock in enumerate(df_shocks.data):
        date_before_shock = date_of_shock - timedelta(minutes=window_width)
        date_after_shock = date_of_shock + timedelta(minutes=window_width)
        range_to_add = pd.date_range(date_before_shock, date_after_shock, freq=date_after_shock - date_before_shock)
        if previous_date_range is None:
            times_range_shocks.append(range_to_add)
            previous_date_range = range_to_add
        else:
            date_ranges_overlap = max(previous_date_range[0], previous_date_range[1]) < min(range_to_add[0],
                                                                                            range_to_add[1])
            if not date_ranges_overlap:
                date_new_range = pd.date_range(previous_date_range[0], range_to_add[1],
                                               freq=range_to_add[1] - previous_date_range[0])
                del times_range_shocks[-1]
                times_range_shocks.append(date_new_range)
                previous_date_range = date_new_range
            else:
                previous_date_range = range_to_add
                times_range_shocks.append(range_to_add)
    union_times_shocks = times_range_shocks[0].union_many(times_range_shocks[1:])
    union_times_shocks = union_times_shocks.tz_localize(None)
    return union_times_shocks


def get_dataset_concatenated(df_feature, union_times_shocks):
    dataset_concatenated = list()
    union_times_shocks = union_times_shocks.tz_localize(None)
    for i in range(0, len(union_times_shocks) - 1, 2):
        window_start_shock = union_times_shocks[i]
        window_end_shock = union_times_shocks[i + 1]

        df_i = df_feature.loc[window_start_shock: window_end_shock]
        dataset_concatenated.append(df_i)

    return pd.concat(dataset_concatenated)


def get_labels(df_dataset, df_shocks, nb_neighbors=0):
    list_labels = [int(0) for x in range(len(df_dataset))]
    for i_shock, date_of_shock in enumerate(df_shocks.data):
        date_of_shock = date_of_shock.replace(tzinfo=None)
        position_shock = df_dataset.index.searchsorted(date_of_shock)
        list_labels[position_shock] = int(1)
        for i_neighbor in range(1, nb_neighbors // 2):
            if position_shock - i_neighbor > 0:
                list_labels[position_shock - i_neighbor] = int(1)
            if position_shock + i_neighbor < len(list_labels):
                list_labels[position_shock + i_neighbor] = int(1)
    return list_labels


def make_model(train_features_shape, metrics, output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    model_ = keras.Sequential([
        keras.layers.Dense(
            60, activation='relu',
            input_shape=(train_features_shape,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid',
                           bias_initializer=output_bias),
    ])

    model_.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model_


def plot_metrics(history, colors_list):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors_list[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors_list[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Shocks Detected (True Negatives): ', cm[0][0])
    print('Legitimate Shocks Incorrectly Detected (False Positives): ', cm[0][1])
    print('Shocks Missed (False Negatives): ', cm[1][0])
    print('Shocks Detected (True Positives): ', cm[1][1])
    print('Total Shocks Detected: ', np.sum(cm[1]))


def split_dataframe(df, chunk_size):
    """
    Split a given dataframe into multiples dataframes with the size of chunk_size
    @param df: the dataframe to be splitted
    @param chunk_size: the size of produced dataframes
    @return: list of dataframes
    """
    chunks = []
    num_chunks = math.ceil(len(df) / chunk_size)
    for i in range(num_chunks):
        chunks.append(df.iloc[i * chunk_size:(i + 1) * chunk_size, :])
    return chunks
