import numpy as np
import tensorflow as tf


def derive_min_max_values():
    mean = [103.939, 116.779, 123.68]
    return 0 - np.array(mean), 255 - np.array(mean)


def find_root_point(x, model, class_idx, steps=200, lr=0.5, stop_threshold=1e-3, **kwargs):
    x_root = tf.Variable(x, trainable=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for step in range(steps):
        with tf.GradientTape() as tape:
            pred = model(x_root)
            loss = pred[:, class_idx]
        grads = tape.gradient(loss, x_root)
        optimizer.apply_gradients([(grads, x_root)])

        # Clip values to stay in valid input range
        clipped = x_root.numpy()
        vmin, vmax = derive_min_max_values()
        for i in range(clipped.shape[-1]):
            clipped[..., i] = np.clip(clipped[..., i], vmin[i], vmax[i])
        x_root.assign(clipped)

        logit_val = loss.numpy()[0]
        print(f"Step {step}, logit = {logit_val:.4f}")
        if logit_val < stop_threshold:
            print("Early stopping: logit near zero.")
            break

    return x_root