import tensorflow as tf

def preprocess(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=3)
    img.set_shape([105, 105, 3])
    img = tf.cast(img, tf.float32) / 255.0
    return img

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.03)
    image = tf.image.random_jpeg_quality(image, 90, 100)
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_twin_train(input_img_path, validation_img_path, label):
    label = tf.cast(label, tf.float32)
    label = tf.expand_dims(label, axis=-1)
    input_img = augment_image(preprocess(input_img_path))
    validation_img = augment_image(preprocess(validation_img_path))
    return (input_img, validation_img, label)

def preprocess_twin_val(input_img_path, validation_img_path, label):
    label = tf.cast(label, tf.float32)
    label = tf.expand_dims(label, axis=-1)
    input_img = preprocess(input_img_path)
    validation_img = preprocess(validation_img_path)
    return (input_img, validation_img, label)
