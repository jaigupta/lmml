from absl.flags import FLAGS
import gin
import tensorflow as tf

from lmmlscripts.core import dataset


def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 0, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 0, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    if indexes.size() != 0:
        y_true_out = tf.tensor_scatter_nd_update(
            y_true_out, indexes.stack(), updates.stack())
    return y_true_out


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-
                        1)  # pylint: disable=no-value-for-parameter, unexpected-keyword-arg

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


_VOC_DS_PREFIXES = ('tfds://coco/', 'tfds://voc/')


def create_voc_mapper(image_size):

    @tf.function
    def voc_mapper(example):
        image = example['image']
        image = tf.image.resize(image, (image_size, image_size)) / 255.
        labels = example['objects']['label']
        labels = tf.expand_dims(tf.cast(labels, tf.float32), axis=-1)
        bboxes = example['objects']['bbox']
        res = tf.concat([bboxes, labels],
                        axis=-1,)  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        return image, tf.sparse.from_dense(res)
    return voc_mapper


def create_waymo_mapper(image_size, image_key):

    @tf.function
    def waymo_mapper(example):
        ex = example[image_key]
        image = ex['image']
        image = tf.image.resize(image, (image_size, image_size)) / 255.

        labels = ex['labels']['type']
        labels = tf.expand_dims(tf.cast(labels, tf.float32), axis=-1)
        bboxes = ex['labels']['bbox']
        res = tf.concat(
            [bboxes, labels], axis=-1)  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter

        return ex, tf.sparse.from_dense(res)

    return waymo_mapper


def create_waymo_ds_for_key(ds_type, split, image_size, image_key):
    return (
        dataset
        .load_dataset(ds_type, split)
        .map(create_waymo_mapper(image_size, image_key)))


def create_waymo_dataset(ds_type, split, image_size):
    _KEYS = ('camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT',
             'camera_SIDE_LEFT', 'camera_SIDE_RIGHT')
    ds = None
    for key in _KEYS:
        new_ds = create_waymo_ds_for_key(ds_type, split, image_size, key)
        ds = new_ds if ds is None else ds.concat(new_ds)

    return ds


@gin.configurable
def load_dataset(ds_type, split: str, batch_size: int, image_size: int = gin.REQUIRED):
    if ds_type.startswith('tfds://waymo_open_dataset/'):
        ds = create_waymo_dataset(ds_type, split, image_size)
    else:
        ds = dataset.load_dataset(ds_type, split)
        if not ds:
            raise ValueError(f'Dataset {ds_type} not handled.')

        if any(ds_type.startswith(prefix) for prefix in _VOC_DS_PREFIXES):
            ds = ds.map(create_voc_mapper(image_size))
        else:
            raise ValueError('Mapper missing for dataset: ' + ds_type)

    return (
        ds
        .batch(batch_size)
        .map(lambda x, y: (x, tf.sparse.to_dense(y)))
        .prefetch(tf.data.AUTOTUNE))
