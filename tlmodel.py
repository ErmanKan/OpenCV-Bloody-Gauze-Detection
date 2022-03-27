import matplotlib.pyplot as plt
import os
import tensorflow as tf
import matplotlib
import numpy as np

IMAGE_SHAPE = (224,224)

TRAINING_PATH = 'Images/train'
TEST_PATH = 'Images/test'

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255)

train_generator = datagen.flow_from_directory(
    TRAINING_PATH,
    shuffle=True,
    target_size=IMAGE_SHAPE)

test_generator = datagen.flow_from_directory(
    TEST_PATH,
    shuffle=False,
    target_size=IMAGE_SHAPE)


def build_model(num_classes):
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', 
                           input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model(num_classes=12)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
print(model.summary())

EPOCHS = 20
BATCH_SIZE = 32
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=test_generator,
                    validation_steps= test_generator.samples // BATCH_SIZE,
                    verbose=1
                    )


train_loss = history.history['loss']
train_acc = history.history['accuracy']
valid_loss = history.history['val_loss']
valid_acc = history.history['val_accuracy']

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()
    # loss plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()
    
save_plots(train_acc, valid_acc, train_loss, valid_loss)

keras_file = "model.h5"
tf.keras.models.save_model(model, keras_file)
print(train_generator.class_indices)
tf.saved_model.save(model,"E:\Erman\Ders\Tez")

converter = tf.lite.TFLiteConverter.from_saved_model("E:\Erman\Ders\Tez")
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
    
labels = '\n'.join(sorted(train_generator.class_indices.keys()))
with open('labels.txt','w') as f:
    f.write(labels)


