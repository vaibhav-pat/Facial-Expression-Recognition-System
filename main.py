import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

print("Tensorflow version:", tf.__version__)

def build_model() -> Sequential:
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(8, activation='softmax'))

    opt = Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser(description='Train facial expression recognition model')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to dataset directory containing train/ and test/ subfolders')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=48)
    parser.add_argument('--output_json', type=str, default='model.json')
    parser.add_argument('--output_weights', type=str, default='model_weights.h5')
    args = parser.parse_args()

    train_dir = os.path.join(args.dataset_dir, 'train')
    test_dir = os.path.join(args.dataset_dir, 'test')

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Expected 'train' and 'test' subdirectories inside {args.dataset_dir}")

    datagen_train = ImageDataGenerator(horizontal_flip=True)
    train_generator = datagen_train.flow_from_directory(
        train_dir,
        target_size=(args.img_size, args.img_size),
        color_mode="grayscale",
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True
    )

    datagen_validation = ImageDataGenerator(horizontal_flip=True)
    validation_generator = datagen_validation.flow_from_directory(
        test_dir,
        target_size=(args.img_size, args.img_size),
        color_mode="grayscale",
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )

    model = build_model()
    model.summary()

    epochs = args.epochs
    steps_per_epoch = train_generator.n // train_generator.batch_size
    validation_steps = validation_generator.n // validation_generator.batch_size

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001, mode='auto'
    )
    checkpoint = ModelCheckpoint(
        args.output_weights, monitor='val_accuracy', save_weights_only=True, mode='max', verbose=1
    )
    callbacks = [checkpoint, reduce_lr]

    model.fit(
        x=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    model_json = model.to_json()
    model.save_weights(args.output_weights)
    with open(args.output_json, "w") as json_file:
        json_file.write(model_json)

if __name__ == '__main__':
    main()