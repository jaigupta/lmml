from dataclasses import dataclass, field
import itertools
import os
from typing import List

from absl import logging
import gin
import tensorflow as tf

from lmmlscripts.core import files


@gin.configurable
@dataclass
class BaseTrainerConfig:
    train_batch_size: int = field(default=8)
    val_batch_size: int = field(default=8)
    run_eval_every: int = field(default=100)
    num_epochs: int = field(default=5)
    eval_iters: int = field(default=20)
    distributed_training: bool = field(default=False)
    modes: List[str] = field(default=('train', 'eval'))
    learning_rate: float = field(default=1e-3)


class BaseTrainer:
    def __init__(self, config):
        self.config = config

    def start(self):
        files.ensure_dirs_exist([
            self.output_dir,
            os.path.join(self.output_dir, 'saved_model'),
            os.path.join(self.output_dir, 'saved_model/backbone'),
            os.path.join(self.output_dir, 'saved_model/classifier'),
            os.path.join(self.output_dir, 'summary'),
            os.path.join(self.output_dir, 'checkpoint'),
        ])
        self._build()
        for _ in range(self.config.num_epochs):
            self.epoch.assign_add(1)
            if 'train' in self.config.modes:
                with self.train_writer.as_default():
                    self._train()
            else:
                assert 'eval' in self.config.modes or 'dev_eval' in self.config.modes
                self.wait_and_load_next_checkpoint()

            if 'eval' in self.config.modes:
                with self.eval_writer.as_default():
                    self._eval()
            if 'dev_eval' in self.config.modes:
                with self.dev_eval_writer.as_default():
                    self._dev_eval()

    def _build(self):
        self.checkpoints_path = ''
        self._setup_devices_and_strategy()

        self.epoch = tf.Variable(0, dtype=tf.int64)
        self.total_steps = tf.Variable(0, dtype=tf.int64)
        self.train_writer = self._create_summary_writer('train')
        self.eval_writer = self._create_summary_writer('eval')
        self.dev_eval_writer = self._create_summary_writer('dev_eval')
    
        self.ds_train = None
        self.ds_val = None

        self.build()
        self._setup_dataset()
        self._init_from_checkpoints(fail_silent=True)

    def _setup_devices_and_strategy(self):
        pass

    def _create_summary_writer(self, name):
        path = os.path.join(self.output_dir, 'summary')
        files.ensure_dir_exists(path)
        path = os.path.join(path, name)
        files.ensure_dir_exists(path)
        return tf.summary.create_file_writer(path)

    def _setup_dataset(self):
        if self.config.distributed_training:
            assert self.ds_train is not None
            self.ds_train = tf.mirrored_strategy.experimental_distribute_dataset(self.ds_train)
            if self.ds_val:
                self.ds_val = tf.mirrored_strategy.experimental_distribute_dataset(self.ds_val)

        if self.config.run_eval_every > 0:
            # Epoch does not use full dataset. Cache iter so that we don't restart.
            # Assuming that dataset is using .repeat() in these scenarios and will not exhaust.
            self.ds_train_iter = iter(self.ds_train)

        if self.ds_val and self.config.eval_iters > 0:
            self.ds_val_iter = iter(self.ds_val)

    def _init_from_checkpoints(self, fail_silent=False):
        checkpoint_path = tf.train.latest_checkpoint(self.checkpoints_path)
        if not checkpoint_path:
            if fail_silent:
                return
            raise FileNotFoundError('Checkpoint not found.')
        self.checkpointer.restore(checkpoint_path)

    def _run_tf_graph(self, graph_fn, *args):
        if self.config.distributed_training:
            return self.mirrored_strategy.run(graph_fn, *args)
        return graph_fn(*args)

    def build(self):
        pass

    def _train(self):
        if self.config.run_eval_every <= 0:
            # Epoch contains full dataset.
            ds_iter = iter(self.ds_train)
        else:
            # An epoch is defined by number of train iterations.
            logging.info('Running max %s iters for train', self.config.run_eval_every)
            ds_iter = itertools.islice(self.ds_train_iter, 0, self.config.run_eval_every)

        for i, data in enumerate(ds_iter):
            logging.info('train/%s', i)
            self.total_steps.assign_add(1)
            self.train_step_start()

            result = self._run_tf_graph(self.train_step, data)

            self.train_step_end(result)
            self.train_writer.flush()

    def train_step_start(self):
        pass

    def train_step(self, _):
        raise NotImplementedError()

    def train_step_end(self, result):
        pass

    def _eval(self):
        assert self.ds_val is not None

        if self.config.run_eval_every <= 0:
            ds_iter = iter(self.ds_train)
        else:
            ds_iter = itertools.islice(self.ds_train_iter, 0, self.config.run_eval_every)

        self.eval_start()

        for data in ds_iter:
            self._run_tf_graph(self.eval_step, data)

        self.eval_end()
        self.eval_writer.flush()

    def eval_start(self):
        pass

    def eval(self):
        pass

    def _eval_end(self):
        self.checkpointer.save(os.path.join(self.output_dir, 'checkpoint'))
        self.eval_end()

    def eval_end(self):
        pass

    def _dev_eval(self):
        pass