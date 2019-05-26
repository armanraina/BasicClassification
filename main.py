import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

plt.imshow(x_train[0], cmap='binary')
plt.show()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

misses_indices = model.predict_classes(x_test) != y_test

misses = x_test[misses_indices]
correct_labels = y_test[misses_indices]
incorrect_labels = model.predict_classes(misses)
for i in range(0, len(misses)):
    plt.imshow(misses[i], cmap='binary')
    title = 'Correct: ' + str(correct_labels[i]) + '; incorrect: ' + str(incorrect_labels[i])
    plt.title(title)
    plt.show()
