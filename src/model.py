
import tensorflow as tf


def build_model(num_classes, image_size=(160, 160)):

    # Load pretrained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])

    # Build model
    inputs = tf.keras.Input(
        shape=(*image_size, 3)
    )

    x = data_augmentation(inputs)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    return model
