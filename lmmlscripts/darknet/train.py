import sys
sys.path.append('../lmml')

from absl import app
from absl import flags
from absl import logging
from lmml.models.darknet import Darknet
from lmml.models.imgclassifier import image_classifier
from lmmlscripts.darknet import dataset
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

flags.DEFINE_enum('mode', 'graph', ['graph', 'eager_tf', 'eager_fit'], 'Mode for training.')
flags.DEFINE_string('dataset', 'imagenet2012', 'The dataset to train on.')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 1000, 'number of classes in the model')
flags.DEFINE_enum('backbone', 'darknet', ['darknet', 'darknet_tiny'], 'The backbone for classifier')

FLAGS = flags.FLAGS


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    model = image_classifier(FLAGS.num_classes, FLAGS.backbone)

    train_dataset = dataset.load_dataset(FLAGS.dataset, 'train', FLAGS.batch_size, FLAGS.size)
    val_dataset = dataset.load_dataset(FLAGS.dataset, 'validation', FLAGS.batch_size, FLAGS.size)

    optimizer = keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                output_logits = model(images, training=True)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = loss_fn(labels, output_logits)
                total_loss = pred_loss + regularization_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            avg_loss.update_state(total_loss)
            return total_loss, pred_loss
    
        @tf.function
        def val_step(images, labels):
            outputs = model(images)
            regularization_loss = tf.reduce_sum(model.losses)
            pred_loss = loss_fn(labels, output_logits)
            total_loss = pred_loss + regularization_loss

            avg_val_loss.update_state(total_loss)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
               total_loss, pred_loss = train_step(images, labels)
               logging.info(f"{epoch}_train_{batch}, {total_loss.numpy()}, {pred_loss.numpy()}")

            for batch, (images, labels) in enumerate(val_dataset):
                total_loss, pred_loss = val_step(images, labels)
                logging.info(f"{epoch}_val_{batch}, {total_loss.numpy()}, {pred_loss.numpy()}")

            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(f'checkpoints/{FLAGS.backbone}_{epoch}.tf')
    else:
        model.compile(optimizer=optimizer, loss=loss_fn, run_eagerly=(FLAGS.mode == 'eager_fit'))

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint(f'checkpoints/{FLAGS.backbone}' + '_{epoch}.tf', verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]

        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)


if __name__ == '__main__':
    app.run(main)