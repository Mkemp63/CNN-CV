import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import layers, models, callbacks

import config

# GPU stuff, this has to be directly called after the tensorflow import
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Memory issue
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("No GPU found")


def load_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape((train_images.shape[0], config.Image_size, config.Image_size, 1))
    test_images = test_images.reshape((test_images.shape[0], config.Image_size, config.Image_size, 1))

    return train_images, train_labels, test_images, test_labels


def evaluate_model(model, test_images: np.ndarray, test_labels: np.ndarray):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_acc)


def make_model(input_shape: tuple, convs: int, filters: list, ks: list, pool_t: list, pool_s: list,
               dense_layers: int, dense_dim: list,
               activations: list = [], activation: str = 'relu', optimiser: str = 'adam', dropout: bool = False):
    """
    input_shape = tuple of the shape of the inputs
    convs = number of convolutional layers
    filters = list of the dimensionalities of the layers
    ks = kernel sizes
    pool_t = list of pool types (string: "max", "avg", "flat")
    pool_s = list of pool sizes
    dense_layers = number of dense layers (output layer excluded)
    dense_dim = list of dimensionality of the layer, number of neurons
    activations = list of the activation methods
    activation = the standard activation method (="relu")
    optimiser = the optimiser (="adam")
    """
    if len(activations) != (convs + dense_layers):
        print("Activations is not long enough")
        activations = [activation] * (convs + dense_layers)
    model = models.Sequential()
    for i in range(0, convs):
        if i == 0:
            model.add(layers.Conv2D(filters[i], (ks[i], ks[i]), activation=activations[i], input_shape=input_shape))
        else:
            model.add(layers.Conv2D(filters[i], (ks[i], ks[i]), activation=activations[i]))

        # Pooling layer
        if pool_t[i] == "max":
            model.add(layers.MaxPooling2D((pool_s[i], pool_s[i])))
        elif pool_t[i] == "avg":
            model.add(layers.AveragePooling2D((pool_s[i], pool_s[i])))
        elif pool_t[i] == "flat":
            model.add(layers.Flatten())

    for i in range(0, dense_layers):
        model.add(layers.Dense(dense_dim[i], activation=activations[convs + i]))

    if dropout:
        model.add(layers.Dropout(0.5))

    model.add(layers.Dense(10))

    model.compile(optimizer=optimiser,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def fit_model(model, train_images: np.ndarray, train_labels: np.ndarray, val_images: np.ndarray, val_labels: np.ndarray,
              print: bool = False, decay: bool = False):
    if decay:
        lr = callbacks.LearningRateScheduler(lr_decay)
        callbacks_list = [lr]
        history = model.fit(train_images, train_labels, epochs=config.Epochs, validation_data=(val_images, val_labels),
                            batch_size=config.Batch_size, callbacks=callbacks_list)
    else:
        history = model.fit(train_images, train_labels, epochs=config.Epochs, validation_data=(val_images, val_labels),
                            batch_size=config.Batch_size)

    if print:
        evaluate_model(model, val_images, val_labels)
    return history, model


# choice task lr decay by 0.5 every 5 epochs
def lr_decay(epoch):
    initial_lr = 0.001
    drop_lr = 0.5
    epochs_for_drop = 5.0
    lr = initial_lr * math.pow(drop_lr, math.floor((1 + epoch) / epochs_for_drop))
    return lr


def plot_val_train_loss(history, title):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()


def k_fold(model, folds: int, train_images: np.ndarray, train_labels: np.ndarray, optimiser: str = 'adam'):
    avg_score = 0
    kFold = KFold(n_splits=folds, shuffle=False)
    for train_index, val_index in kFold.split(train_images, train_labels):
        tr_imgs, val_imgs = train_images[train_index], train_images[val_index]
        tr_lab, val_lab = train_labels[train_index], train_labels[val_index]

        model_ = models.clone_model(model)
        model_.compile(optimizer=optimiser, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])

        trained_model = fit_model(model_, tr_imgs, tr_lab, val_imgs, val_lab)
        score = model_.evaluate(val_imgs, val_lab)
        # print(f"  Score: {score[1]}")
        avg_score += score[1]
    avg_score = avg_score / folds
    # print(f"Average score: {avg_score}")
    return avg_score

	
def main():
    # data loading
    train_images, train_labels, test_images, test_labels = load_data()
    img_rows = train_images[0].shape[0]
    img_cols = train_images[0].shape[1]
    input_shape = (img_rows, img_cols, 1)

    # testje(train_labels, 5)
    aantal = int(config.Validate_perc * len(train_labels) * -1)
    val_images1 = train_images[aantal:]
    val_labels1 = train_labels[aantal:]
    train_images1 = train_images[:aantal]
    train_labels1 = train_labels[:aantal]

    # model_b = make_model(input_shape, 3, [16, 32, 32], [3, 3, 3], ["max", "max", "flat"], [2, 2, 0], 1, [32])
    # k_fold(model_b, 5, train_images, train_labels)

    # baseline model
    training = not config.Use_pretrained
    if config.Use_pretrained:
        try:
            model_baseline = tf.keras.models.load_model(config.BASELINE_DIR)
        except OSError:
            print("Model baseline doesnt exist")
            training = True
    if training:
        # history_baseline, model_baseline = fit_baseline_model(input_shape, train_images, train_labels, val_images,
        #                                                       val_labels)
        model_baseline = make_model(input_shape, 3, [16, 32, 32], [3, 3, 3], ["max", "max", "flat"], [2, 2, 0], 1, [32])
        history_baseline, model_baseline = fit_model(model_baseline, train_images1, train_labels1, val_images1,
                                                     val_labels1)

        model_baseline.save(config.BASELINE_DIR)
        print(config.BASELINE_DIR)
        plot_val_train_loss(history_baseline, "Baseline model")

    try:
        model_less = tf.keras.models.load_model('./models/model_less/')
    except OSError:
        model_less = make_model(input_shape, 3, [16, 32, 32], [3, 4, 3], ["max", "max", "flat"], [2, 2, 0], 1, [32])
        history_less, model_less = fit_model(model_less, train_images1, train_labels1, val_images1, val_labels1)

        model_less.save('./models/model_less/')
        plot_val_train_loss(history_less, "model less")

    try:
        model_bigger_nn = tf.keras.models.load_model('./models/model_bigger_nn/')
    except OSError:
        model_bigger_nn = make_model(input_shape, 3, [16, 32, 32], [3, 3, 3], ["max", "max", "flat"], [2, 2, 0], 1, [48])
        history_bigger_nn, model_bigger_nn = fit_model(model_bigger_nn, train_images1, train_labels1, val_images1, val_labels1)

        model_bigger_nn.save('./models/model_bigger_nn/')
        plot_val_train_loss(history_bigger_nn, "model bigger nn")

    #
    # NOG AAN TE PASSEN
    #
    # other model
    # dropout model
    try:
        model_dropout = tf.keras.models.load_model(config.DROPOUT_DIR)
    except OSError:
        print("Model dropout doesnt exist")
        model_dropout = make_model(input_shape, 3, [16, 32, 32], [3, 3, 3], ["max", "max", "flat"],
                                   [2, 2, 0], 1, [32], dropout=True)

        history_dropout, model_dropout = fit_model(model_dropout, train_images1, train_labels1, val_images1,
                                                   val_labels1)
        model_dropout.save(config.DROPOUT_DIR)
        plot_val_train_loss(history_dropout, "Dropout model")

    # lr decay model (choice task)
    try:
        model_lr_decay = tf.keras.models.load_model(config.DECAY_DIR)
    except OSError:
        print("Model decay doesnt exist")
        model_lr_decay = make_model(input_shape, 3, [16, 32, 32], [3, 3, 3], ["max", "max", "flat"],
                                    [2, 2, 0], 1, [32])

        history_lr_decay, model_lr_decay = fit_model(model_lr_decay, train_images1, train_labels1, val_images1,
                                                     val_labels1, decay=True)
        model_dropout.save(config.DECAY_DIR)
        plot_val_train_loss(history_lr_decay, "Decay model")


if __name__ == '__main__':
    main()
