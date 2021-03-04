import attr
import itertools
import os
import time
from typing import List

from absl import logging
import gin
import gin.tf.external_configurables
import tensorflow as tf

from lmmlscripts.core import files


class DummyScope:
    def __enter__(self, *args):
        pass

    def __exit__(self, *args):
        pass


class DummyStrategy:
    def scope(self):
        return DummyScope()


def split_cpu_to_multiple_virtual_devices():
    cpu = tf.config.list_physical_devices('CPU')[0]
    tf.config.set_logical_device_configuration(
        cpu,
        [tf.config.LogicalDeviceConfiguration(),
        tf.config.LogicalDeviceConfiguration()])


def _try_setup_tpu_strategy():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        topology = tf.tpu.experimental.initialize_tpu_system(resolver)
        logging.info('topology.mesh_shape: %s', topology.mesh_shape)
        logging.info('topology._device_coordinates: %s',
                     topology.device_coordinates)
        return tf.distribute.experimental.TPUStrategy(resolver)
    except ValueError as e:
        logging.warn('Could not initialize TPU: %s', e)
        return None


def _setup_devices():
    tpu_devices = tf.config.experimental.list_logical_devices('TPU')
    if tpu_devices:
        return tpu_devices

    gpu_devices = tf.config.experimental.list_logical_devices('GPU')
    if gpu_devices:
        return gpu_devices

    return tf.config.experimental.list_logical_devices('CPU')


def _setup_devices_and_strategy(distributed_training: bool):
    tpu_strategy = _try_setup_tpu_strategy() if distributed_training else DummyStrategy()
    devices = _setup_devices()
    if tpu_strategy:
        return devices, tpu_strategy
    return devices, tf.distribute.MirroredStrategy(devices=devices) if distributed_training else DummyStrategy()


def _check_has_obj(o, expected_type):
    if isinstance(o, tuple) or isinstance(o, list):
        return any(_check_has_obj(oi, expected_type) for oi in o)
    return isinstance(o, expected_type)


def _validate_checkpointer(checkpointer: tf.train.Checkpoint):
    assert checkpointer is not None
    objs = {o.name: o.ref for o in checkpointer._checkpoint_dependencies}
    assert 'epoch' in objs
    assert 'total_steps' in objs
    if not any(_check_has_obj(o, tf.keras.Model) for o in objs.items()):
        logging.warn('No keras model in checkpointer.')
    if not any(_check_has_obj(o, tf.keras.optimizers.Optimizer) for o in objs.items()):
        logging.warn('No keras optimizer in checkpointer.')


def _save_gin_config(output_dir: str):
    # Replace \n with \n\n to improve logging.
    gin_config = gin.operative_config_str().replace('\n', '\n\n')
    logging.info('gin config: %s', gin_config)
    tf.summary.text('gin_config', gin_config, 0)
    with tf.io.gfile.GFile(os.path.join(output_dir, 'gin_config.txt'), 'w') as f:
        f.write(gin_config)


@gin.configurable
@attr.s
class BaseTrainerConfig:
    train_batch_size: int = attr.ib(default=8)
    val_batch_size: int = attr.ib(default=8)
    run_eval_every: int = attr.ib(default=100)
    num_epochs: int = attr.ib(default=5)
    eval_iters: int = attr.ib(default=20)
    distributed_training: bool = attr.ib(default=False)
    modes: List[str] = attr.ib(default=('train', 'eval'))
    learning_rate: float = attr.ib(default=1e-3)
    initialization_checkpoint: str = attr.ib(default=None)


class BaseTrainer:
    def __init__(self, output_dir: str):
        self.config = BaseTrainerConfig()
        self.output_dir = output_dir

    def start(self):
        files.ensure_dirs_exist([
            self.output_dir,
            os.path.join(self.output_dir, 'saved_model'),
            os.path.join(self.output_dir, 'saved_model/backbone'),
            os.path.join(self.output_dir, 'saved_model/classifier'),
            os.path.join(self.output_dir, 'summary'),
            os.path.join(self.output_dir, 'ckpts'),
        ])
        self._build()
        while self.epoch.numpy() < self.config.num_epochs:
            self.epoch.assign_add(1)
            if 'train' in self.config.modes:
                with self.train_writer.as_default():  # pylint: disable=not-context-manager
                    self._train()
            else:
                assert 'eval' in self.config.modes or 'dev_eval' in self.config.modes
                self._wait_and_load_next_checkpoint()

            if 'eval' in self.config.modes:
                with self.eval_writer.as_default():  # pylint: disable=not-context-manager
                    self._eval()
            if 'dev_eval' in self.config.modes:
                with self.dev_eval_writer.as_default():  # pylint: disable=not-context-manager
                    self._dev_eval()

    def _build(self):
        self.checkpoints_path = os.path.join(self.output_dir, 'ckpts')
        self.devices, self.mirrored_strategy = _setup_devices_and_strategy(
            self.config.distributed_training)
        tf.debugging.set_log_device_placement(True)

        self.epoch = tf.Variable(0, dtype=tf.int64)
        self.total_steps = tf.Variable(0, dtype=tf.int64)
        if 'train' in self.config.modes:
            self.train_writer = self._create_summary_writer('train')
            with self.train_writer.as_default():
                _save_gin_config(self.output_dir)

        if 'eval' in self.config.modes:
            self.eval_writer = self._create_summary_writer('eval')
        if 'dev_eval' in self.config.modes:
            self.dev_eval_writer = self._create_summary_writer('dev_eval')

        self.ds_train = None
        self.ds_val = None
        self.checkpointer = None

        self.build()
        _validate_checkpointer(self.checkpointer)
        self._setup_dataset()
        if 'train' in self.config.modes:
            # evals load later, in _wait_and_load_next_checkpoint.
            # Hence, loading only for 'train' mode.
            self._init_from_checkpoints(
                initialization_path=self.config.initialization_checkpoint,
                fail_silent=self.config.initialization_checkpoint is None)

    def _wait_and_load_next_checkpoint(self):
        cur_total_steps = self.total_steps.numpy()
        logging.warn('waiting for next checkpoint steps: %s', cur_total_steps)
        while True:
            self._init_from_checkpoints()
            if cur_total_steps != self.total_steps.numpy():
                break
            logging.warn('waiting for next checkpoint steps: %s',
                         cur_total_steps)
            time.sleep(10)

    def _create_summary_writer(self, name):
        path = os.path.join(self.output_dir, 'summary')
        files.ensure_dir_exists(path)
        path = os.path.join(path, name)
        files.ensure_dir_exists(path)
        return tf.summary.create_file_writer(path)

    def _setup_dataset(self):
        if self.config.distributed_training:
            assert self.ds_train is not None
            self.ds_train = self.mirrored_strategy.experimental_distribute_dataset(
                self.ds_train)
            if self.ds_val:
                self.ds_val = self.mirrored_strategy.experimental_distribute_dataset(
                    self.ds_val)

        if self.config.run_eval_every > 0:
            # Epoch does not use full dataset. Cache iter so that we don't restart.
            # Assuming that dataset is using .repeat() in these scenarios and will not exhaust.
            self.ds_train_iter = iter(self.ds_train)

        if self.ds_val and self.config.eval_iters > 0:
            self.ds_val_iter = iter(self.ds_val)

    def _init_from_checkpoints(self, initialization_path=None, fail_silent=False):
        if initialization_path:
            checkpoint_path = (
                tf.train.latest_checkpoint(initialization_path)
                if tf.io.gfile.isdir(initialization_path) else initialization_path)
        else:
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoints_path)
        if not checkpoint_path:
            if fail_silent:
                return
            raise FileNotFoundError('Checkpoint not found.')
        with self.mirrored_strategy.scope():
            self.checkpointer.restore(checkpoint_path)

    def _run_tf_graph(self, graph_fn, *args):
        if self.config.distributed_training:
            return self.mirrored_strategy.run(graph_fn, args)
        return graph_fn(*args)

    def build(self):
        pass

    def _train(self):
        if self.config.run_eval_every <= 0:
            # Epoch contains full dataset.
            ds_iter = iter(self.ds_train)
        else:
            # An epoch is defined by number of train iterations.
            logging.info('Running max %s iters for train',
                         self.config.run_eval_every)
            ds_iter = itertools.islice(
                self.ds_train_iter, 0, self.config.run_eval_every)

        for batch in ds_iter:
            step = self.total_steps.numpy()
            if step < 100 or step % 100 == 0:
                logging.info('train/%s/%s', self.epoch.numpy(),
                             self.total_steps.numpy())
            self.total_steps.assign_add(1)
            self.train_step_start()

            if self.config.distributed_training:
                result = self.mirrored_strategy.run(self.train_step, batch)
            else:
                result = self.train_step(batch)
            self.train_step_end(*(result or ()))

            self.train_writer.flush()

        self._train_epoch_end()

    def train_step_start(self):
        pass

    def train_step(self, _):
        raise NotImplementedError()

    def train_step_end(self, *args):
        pass

    def _train_epoch_end(self):
        self.checkpointer.save(os.path.join(self.checkpoints_path, 'ckpt'))
        self.train_epoch_end()

    def train_epoch_end(self):
        pass

    def _eval(self):
        assert self.ds_val is not None

        if self.config.run_eval_every <= 0:
            ds_iter = iter(self.ds_val)
        else:
            ds_iter = itertools.islice(
                self.ds_val_iter, 0, self.config.eval_iters)

        logging.info('Eval/%s', self.epoch.numpy())
        self.eval_start()

        for batch in ds_iter:
            if self.config.distributed_training:
                self.mirrored_strategy.run(self.eval_step, batch)
            else:
                self.eval_step(batch)

        self.eval_end()
        self.eval_writer.flush()

    def eval_start(self):
        pass

    def eval_step(self, batch):
        pass

    def _eval_end(self):
        self.eval_end()

    def eval_end(self):
        pass

    def _dev_eval(self):
        pass
