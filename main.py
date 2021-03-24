import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import config
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold


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
               activations: list = [], activation: str = 'relu', optimiser: str = 'adam'):
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
    if len(activations) != (convs+dense_layers):
        print("Activations is not long enough")
        activations = [activation] * (convs+dense_layers)
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
        model.add(layers.Dense(dense_dim[i], activation=activations[convs+i]))

    model.add(layers.Dense(10))

    model.compile(optimizer=optimiser,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def fit_model(model, train_images: np.ndarray, train_labels: np.ndarray, val_images: np.ndarray, val_labels: np.ndarray,
              print: bool = False):
    history = model.fit(train_images, train_labels, epochs=config.Epochs, validation_data=(val_images, val_labels),
                        batch_size=config.Batch_size)
    if print:
        evaluate_model(model, val_images, val_labels)
    return history, model


def fit_baseline_model(input_shape: tuple, train_images: np.ndarray, train_labels: np.ndarray, test_images: np.ndarray,
                       test_labels: np.ndarray):
    model = models.Sequential()

    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print(model.summary())
    history = model.fit(train_images, train_labels, epochs=config.Epochs, validation_data=(test_images, test_labels),
                        batch_size=config.Batch_size)
    evaluate_model(model, test_images, test_labels)
    return history, model


def plot_val_train_loss(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


def testje(train_labels, aantal):
    grootte = int(len(train_labels)/aantal)
    for i in range(0, aantal):
        geteld = [0] * 10
        startIndex = i * grootte
        for j in range(0, grootte):
            geteld[train_labels[startIndex+j]] += 1
        ans = "Totals: "
        for k in range(0, 10):
            ans += f"{k}: {geteld[k]}, "
        print(ans)


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
    avg_score = avg_score/folds
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
    input()

    # baseline model
    training = not config.Use_pretrained
    if config.Use_pretrained:
        try:
            model_baseline = tf.keras.models.load_model(config.BASELINE_DIR)
        except OSError:
            print("Model doesnt exist")
            training = True
    if training:
        # history_baseline, model_baseline = fit_baseline_model(input_shape, train_images, train_labels, val_images,
        #                                                       val_labels)
        model_baseline = make_model(input_shape, 3, [16, 32, 32], [3, 3, 3], ["max", "max", "flat"], [2, 2, 0], 1, [32])
        history_baseline, model_baseline = fit_model(model_baseline, train_images1, train_labels1, val_images1,
                                                     val_labels1)

        model_baseline.save(config.BASELINE_DIR)
        print(config.BASELINE_DIR)
        plot_val_train_loss(history_baseline)

    # other model

    # other model


if __name__ == '__main__':
    main()
