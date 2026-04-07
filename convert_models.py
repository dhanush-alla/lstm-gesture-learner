"""
convert_models.py
Rebuilds model_asl.keras / model_isl.keras (saved with Keras 3 / mixed_float16)
into plain Keras 2 .h5 files that TF 2.15 can load without issues.

Run once:  python convert_models.py
"""

import os, zipfile, tempfile
import numpy as np
import h5py
import tensorflow as tf

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def _build_model(num_classes: int) -> 'tf.keras.Model':
    tf.keras.backend.clear_session()   # reset naming counters so dense/bn names start from 0
    inp = tf.keras.Input(shape=(63,))
    x   = tf.keras.layers.Dense(256, activation='relu')(inp)
    x   = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(128, activation='relu')(x)
    x   = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(64, activation='relu')(x)
    x   = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    x   = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inp, out)
    model(tf.zeros((1, 63)), training=False)
    return model


def _transfer_weights(model: 'tf.keras.Model', keras_zip: str) -> None:
    """Read weights from Keras-3 model.weights.h5 and assign to Keras-2 model."""
    with zipfile.ZipFile(keras_zip) as z:
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(z.read('model.weights.h5'))

    assigned = skipped = 0
    try:
        with h5py.File(tmp_path, 'r') as f:
            layers_grp = f['layers']
            for layer in model.layers:
                if layer.name not in layers_grp:
                    continue
                layer_grp = layers_grp[layer.name]
                if 'vars' not in layer_grp:
                    continue
                vars_grp  = layer_grp['vars']
                var_keys  = sorted(vars_grp.keys(), key=lambda k: int(k))
                weights_np = [vars_grp[k][()] for k in var_keys]
                cur_weights = layer.get_weights()
                if len(weights_np) != len(cur_weights):
                    print(f"  [warn] {layer.name}: expected {len(cur_weights)} vars, got {len(weights_np)} -- skipping")
                    skipped += 1
                    continue
                weights_np = [w.astype(np.float32) for w in weights_np]
                ok = all(w.shape == c.shape for w, c in zip(weights_np, cur_weights))
                if not ok:
                    print(f"  [warn] {layer.name}: shape mismatch -- skipping")
                    skipped += 1
                    continue
                layer.set_weights(weights_np)
                assigned += 1
    finally:
        os.unlink(tmp_path)

    print(f"  weights transferred: {assigned} layers assigned, {skipped} skipped")


def convert(src_name: str, num_classes: int) -> None:
    src = os.path.join(MODELS_DIR, src_name)
    dst = os.path.join(MODELS_DIR, src_name.replace('.keras', '_compat.h5'))

    print(f"Converting {src_name} ({num_classes} classes)...")
    model = _build_model(num_classes)
    _transfer_weights(model, src)
    model.save(dst)
    print(f"  -> saved {os.path.basename(dst)}")

    loaded = tf.keras.models.load_model(dst, compile=False)
    d1 = np.zeros((1, 63), dtype=np.float32); d1[0, 0] = 1.0
    d2 = np.zeros((1, 63), dtype=np.float32); d2[0, 0] = 0.5
    o1 = loaded(tf.constant(d1), training=False).numpy()
    o2 = loaded(tf.constant(d2), training=False).numpy()
    changed = not np.allclose(o1, o2, atol=1e-4)
    print(f"  -> reload OK  shape={o1.shape}  predictions vary={changed}")
    if not changed:
        print("  [WARN] outputs identical -- weights may not have loaded!")
    print()


if __name__ == '__main__':
    convert('model_asl.keras', num_classes=29)
    convert('model_isl.keras', num_classes=26)
    print("Done.")

