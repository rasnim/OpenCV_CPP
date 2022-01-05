import tensorflow as tf

from tensorflow.keras import datasets, layers, models

def cnn_model(learning_rate):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate),
                  loss=tf.nn.softmax_cross_entropy_with_logits,
                  metrics=['accuracy'])

    model.summary()

    return model


if __name__ == '__main__':
    learning_rate = 0.001
    training_epochs = 20
    batch_size = 100

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32')
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32')

    # 픽셀 값을 0~1 사이로 정규화합니다.
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print('Start learning!')

    model = cnn_model(learning_rate)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

    test_loss, test_acc = model.evaluate(x_test, y_test)

    print("\n Test Accuracy: %.4f" % test_acc)
    print('Learning finished!')