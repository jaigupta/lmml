import itertools
import os

from lmmlscripts.core import trainer
from lmmlscripts.yolo import dataset
from lmml.models.yolov3 import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks,
    freeze_all
)
from lmml.core.oop import overrides
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
import tensorflow as tf
import numpy as np
import gin
from absl.flags import FLAGS
from absl import app, flags, logging

flags.DEFINE_multi_string(
    'gin_param', None, 'Repeated Gin parameter bindings.')
flags.DEFINE_string('model_config', 'base', 'Repeated Gin parameter bindings.')


@gin.configurable
class Trainer(trainer.BaseTrainer):

    def __init__(
            self,
            image_size=gin.REQUIRED,
            backbone_path=gin.REQUIRED,
            num_classes=gin.REQUIRED,
            dataset=gin.REQUIRED,
            output_dir=gin.REQUIRED,
            dev_eval_batch_size=2,
            dev_eval_iters=100):
        super().__init__(output_dir)
        self.image_size = image_size
        self.backbone_path = backbone_path
        self.num_classes = num_classes
        self.dataset = dataset
        self.dev_eval_batch_size = dev_eval_batch_size
        self.dev_eval_iters = dev_eval_iters

    @overrides(trainer.BaseTrainer)
    def build(self):
        self.anchors = yolo_anchors
        self.anchor_masks = yolo_anchor_masks
        with self.mirrored_strategy.scope():
            self.backbone_model = tf.saved_model.load(
                self.backbone_path).signatures['serving_default']

        fake_images = tf.zeros((8, self.image_size, self.image_size, 3))
        outputs = self.backbone_model(fake_images)
        x_36, x_61, x = outputs['add_10'], outputs['add_18'], outputs['add_22']

        with self.mirrored_strategy.scope():
            self.model = YoloV3(
                (x_36.shape[1:], x_61.shape[1:], x.shape[1:]),
                training=True,
                classes=self.num_classes)

            self.optimizer = tf.keras.optimizers.Adam(
                lr=self.config.learning_rate)
            self.loss_fns = [
                YoloLoss(tf.gather(self.anchors, mask),
                         classes=self.num_classes)
                for mask in self.anchor_masks]

        self.ds_train = dataset.load_dataset(
            self.dataset, 'train', self.config.train_batch_size, self.image_size)
        self.ds_val = dataset.load_dataset(
            self.dataset, 'validation', self.config.val_batch_size, self.image_size)

        self.avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        self.checkpointer = tf.train.Checkpoint(
            model=self.model,
            # backbone_model=self.backbone_model,
            optimizer=self.optimizer,
            total_steps=self.total_steps,
            epoch=self.epoch)

        self.ds_dev = dataset.load_dataset(
            self.dataset, 'validation', self.dev_val_batch_size, self.image_size)
        if self.config.distributed_training:
            self.ds_dev = self.mirrored_strategy.experimental_distribute_dataset(self.ds_dev)
            self.ds_dev_iter = iter(self.ds_dev)

    @overrides(trainer.BaseTrainer)
    @tf.function(experimental_relax_shapes=True)
    def train_step(self, images, labels):
        transformed_labels = dataset.transform_targets(
            labels, self.anchors, self.anchor_masks, self.image_size)
        features = self.backbone_model(images)  # TODO: training = False
        f_36, f_61, f = features['add_10'], features['add_18'], features['add_22']
        with tf.GradientTape() as tape:
            outputs = self.model((f_36, f_61, f), training=True)
            regularization_loss = tf.reduce_sum(self.model.losses)
            pred_loss = []
            for output, label, loss_fn in zip(outputs, transformed_labels, self.loss_fns):
                pred_loss.append(loss_fn(label, output))
            total_loss = tf.reduce_sum(pred_loss) + regularization_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        self.avg_loss.update_state(total_loss)
        return total_loss, tuple(tf.reduce_mean(l) for l in pred_loss)

    @overrides(trainer.BaseTrainer)
    def train_step_end(self, total_loss, pred_loss):
        def replica_avg(vals):
            return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, vals, axis=None)

        tf.summary.scalar('epoch', self.epoch, self.total_steps)
        tf.summary.scalar('loss/avg', self.avg_loss.result(), self.total_steps)
        tf.summary.scalar(
            'loss/total', replica_avg(total_loss), self.total_steps)
        tf.summary.scalar(
            'loss/pred0', replica_avg(pred_loss[0]), self.total_steps)
        tf.summary.scalar(
            'loss/pred1', replica_avg(pred_loss[1]), self.total_steps)
        tf.summary.scalar(
            'loss/pred2', replica_avg(pred_loss[2]), self.total_steps)

    @overrides(trainer.BaseTrainer)
    def train_epoch_end(self):
        self.avg_loss.reset_states()
        tf.saved_model.save(
            self.model,
            os.path.join(self.output_dir, 'saved_model/model'))

    @overrides(trainer.BaseTrainer)
    @tf.function(experimental_relax_shapes=True)
    def eval_step(self, images, labels):
        transformed_labels = dataset.transform_targets(
            labels, self.anchors, self.anchor_masks, self.image_size)
        features = self.backbone_model(images)
        f_36, f_61, f = features['add_10'], features['add_18'], features['add_22']
        outputs = self.model((f_36, f_61, f), training=False)
        regularization_loss = tf.reduce_sum(self.model.losses)
        pred_loss = []
        for output, label, loss_fn in zip(outputs, transformed_labels, self.loss_fns):
            pred_loss.append(loss_fn(label, output))
        total_loss = tf.reduce_sum(pred_loss) + regularization_loss

        self.avg_val_loss.update_state(total_loss)

    @overrides(trainer.BaseTrainer)
    def eval_end(self):
        tf.summary.scalar(
            'loss/avg', self.avg_val_loss.result(), self.total_steps)

        self.avg_val_loss.reset_states()

    @overrides(trainer.BaseTrainer)
    def dev_eval(self):
        for i, batch in enumerate(itertools.islice(self.ds_dev_iter, 0, self.dev_eval_iters)):
            images, labels = batch
            transformed_labels = dataset.transform_targets(
                labels, self.anchors, self.anchor_masks, self.image_size)
            features = self.backbone_model(images)
            f_36, f_61, f = features['add_10'], features['add_18'], features['add_22']
            outputs = self.model((f_36, f_61, f), training=False)
            # TODO: Plot image with actual labels
            # TODO: Plot generated outputs on the same image with a different color.


def main(_argv):
    gin.parse_config_files_and_bindings(
        [f'lmmlscripts/gin/yolo/{flags.FLAGS.model_config}.gin'],
        flags.FLAGS.gin_param)

    Trainer().start()


if __name__ == '__main__':
    app.run(main)
