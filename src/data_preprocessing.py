
import tensorflow as tf
from pathlib import Path


def load_datasets(
    dataset_path,
    image_size=(160, 160),
    batch_size=32
):
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    # 70% Training Dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.30,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int"
    )

    # Remaining 30%
    temp_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.30,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int"
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)

    # Split remaining data into validation and test
    total_batches = int(
        tf.data.experimental.cardinality(temp_dataset).numpy()
    )

    val_batches = total_batches // 2

    validation_dataset = temp_dataset.take(val_batches)
    test_dataset = temp_dataset.skip(val_batches)

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.shuffle(
        500, seed=42
    ).prefetch(AUTOTUNE)

    validation_dataset = validation_dataset.prefetch(AUTOTUNE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names,
        num_classes
    )
