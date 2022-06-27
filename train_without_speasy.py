from pickle import dump

import numpy as np
import pandas as pd
from matplotlib import colors, pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from utils import make_model, plot_metrics

# Column's name of the shock in dataframe
EVENT_LABEL = "event"

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
    data_set_for_training_1 = pd.read_pickle("dataset_1_15.pkl")
    data_set_for_training_2 = pd.read_pickle("dataset_16_29.pkl")
    data_set_for_training = data_set_for_training_1.append(data_set_for_training_2)
    print(data_set_for_training)
    print(data_set_for_training.info())

    neg, pos = np.bincount(data_set_for_training[EVENT_LABEL])
    total = neg + pos
    print('Class imbalanced:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

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

    #model.save('model.h5')
    #dump(scaler, open('scaler.pkl', 'wb'))

    plot_metrics(baseline_history, colors_list)
    plt.show()

