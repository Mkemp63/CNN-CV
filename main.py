import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

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
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    return train_images, train_labels, test_images, test_labels


def fit_baseline_model(input_shape, train_images, train_labels, test_images, test_labels):
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
    history = model.fit(train_images, train_labels, epochs=12, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_acc)
    return history, model


def plot_val_train_loss(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


def main():
    # data loading
    train_images, train_labels, test_images, test_labels = load_data()
    img_rows = train_images[0].shape[0]
    img_cols = test_images[0].shape[1]
    input_shape = (img_rows, img_cols, 1)

    # baseline model
    try:
        model_baseline = tf.keras.models.load_model('./models/model_baseline/')
    except OSError:
        print("Model doesnt exist")
        history_baseline, model_baseline = fit_baseline_model(input_shape, train_images, train_labels, test_images,
                                                              test_labels)
        model_baseline.save('./models/model_baseline/')
        plot_val_train_loss(history_baseline)

    # other model

    # other model


if __name__ == '__main__':
    main()
