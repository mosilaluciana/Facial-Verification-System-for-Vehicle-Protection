import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from datetime import datetime
import warnings
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from scripts.data_loader import get_people_dict, get_negative_list, make_tf_dataset
from models.siamese_model import make_siamese_model
from utils.preprocessing import preprocess


def smart_data_strategy(people_dict, negative_images):
    total_people = len(people_dict)
    total_images = sum(len(images) for images in people_dict.values())
    avg_images_per_person = total_images / total_people if total_people > 0 else 0

    if avg_images_per_person >= 8:
        pairs_per_person = min(80, int(avg_images_per_person * 7))
    elif avg_images_per_person >= 5:
        pairs_per_person = min(60, int(avg_images_per_person * 9))
    else:
        pairs_per_person = min(40, int(avg_images_per_person * 12))

    positive_pairs = []
    negative_pairs = []

    for person, images in people_dict.items():
        if len(images) >= 3:
            from itertools import combinations
            all_combinations = list(combinations(images, 2))
            sampled_combinations = random.sample(
                all_combinations,
                min(len(all_combinations), pairs_per_person)
            )
            for img1, img2 in sampled_combinations:
                positive_pairs.append((img1, img2, 1))
        elif len(images) == 2:
            for _ in range(pairs_per_person):
                positive_pairs.append((images[0], images[1], 1))
        elif len(images) == 1:
            for _ in range(pairs_per_person // 2):
                positive_pairs.append((images[0], images[0], 1))

    target_negatives = len(positive_pairs)

    if negative_images and len(negative_images) > 10:
        neg_external_count = int(target_negatives * 0.4)
        for _ in range(neg_external_count):
            person = random.choice(list(people_dict.keys()))
            person_img = random.choice(people_dict[person])
            neg_img = random.choice(negative_images)
            negative_pairs.append((person_img, neg_img, 0))

    if total_people >= 2:
        remaining_negatives = target_negatives - len(negative_pairs)
        persons_list = list(people_dict.keys())
        for _ in range(remaining_negatives):
            p1, p2 = random.sample(persons_list, 2)
            img1 = random.choice(people_dict[p1])
            img2 = random.choice(people_dict[p2])
            negative_pairs.append((img1, img2, 0))

    final_count = min(len(positive_pairs), len(negative_pairs))
    balanced_pairs = positive_pairs[:final_count] + negative_pairs[:final_count]
    random.shuffle(balanced_pairs)

    return balanced_pairs


def configure_training_params(num_people, total_images, dataset_size):

    if dataset_size < 100:
        batch_size = 4
        epochs = 5
        learning_rate = 1e-6
        patience = 3
        validation_split = 0.25
    elif dataset_size < 300:
        batch_size = 8
        epochs = 7
        learning_rate = 5e-7
        patience = 4
        validation_split = 0.2
    else:
        batch_size = 16
        epochs = 10
        learning_rate = 2e-7
        patience = 5
        validation_split = 0.2

    if num_people <= 2:
        epochs += 2
        learning_rate *= 1.2
    elif num_people >= 10:
        epochs -= 1
        learning_rate *= 0.8

    return {
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'patience': patience,
        'validation_split': validation_split
    }


def create_callbacks(model_path, params):
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=params['patience'],
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-9,
            verbose=1,
            cooldown=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path.replace('.h5', '_best.h5'),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1,
            save_weights_only=False
        )
    ]


def optimize_threshold(model, val_dataset):
    y_true = []
    y_pred = []

    for batch in val_dataset:
        x, y = batch
        pred = model(x, training=False).numpy().reshape(-1)
        y_true.append(y.numpy())
        y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    best_metrics = {}
    best_f1 = 0
    best_threshold = 0.5

    thresholds = np.arange(0.25, 0.75, 0.005)

    for threshold in thresholds:
        y_pred_bin = (y_pred > threshold).astype(int)

        try:
            accuracy = accuracy_score(y_true, y_pred_bin)
            precision = precision_score(y_true, y_pred_bin, zero_division=0)
            recall = recall_score(y_true, y_pred_bin, zero_division=0)
            f1 = f1_score(y_true, y_pred_bin, zero_division=0)

            combined_score = 0.4 * f1 + 0.3 * accuracy + 0.3 * (precision + recall) / 2

            if combined_score > best_f1:
                best_f1 = combined_score
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'combined_score': combined_score
                }
        except:
            continue

    try:
        auc = roc_auc_score(y_true, y_pred)
        best_metrics['auc'] = auc
    except:
        best_metrics['auc'] = 0.0

    return best_threshold, best_metrics


def optimize_model(model):
    # Pentru fine-tuning adevarat - inghetam majoritatea layerelor
    frozen_layers = 0
    total_layers = len(model.layers)

    for i, layer in enumerate(model.layers):
        # inghetam 80% din layere pentru fine-tuning adevarat
        if i < total_layers * 0.8:
            layer.trainable = False
            frozen_layers += 1
        else:
            layer.trainable = True
            # Doar ultimele layere se antreneaza

    return model


def augment_image(image):
    image = tf.cond(
        tf.random.uniform([]) > 0.5,
        lambda: tf.image.random_flip_left_right(image),
        lambda: image
    )
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_train(img1_path, img2_path, label):
    img1 = preprocess(img1_path)
    img2 = preprocess(img2_path)
    img1 = augment_image(img1)
    img2 = augment_image(img2)
    return (img1, img2), tf.cast(label, tf.float32)


def preprocess_val(img1_path, img2_path, label):
    img1 = preprocess(img1_path)
    img2 = preprocess(img2_path)
    return (img1, img2), tf.cast(label, tf.float32)


def main():
    # Paths
    app_user_images = os.path.abspath("../APLICATIE/user_images/cropped_images")
    source_negatives = os.path.abspath("./data/negatives_fine_tune")
    base_model_path = os.path.abspath("siamese_face_recognition_model.h5")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = r"E:\LICENTA\PROIECT  APLICATIE MASINA+FV\APLICATIE\model"
    os.makedirs(output_dir, exist_ok=True)
    new_model_path = os.path.join(output_dir, f"siamese_optimized_{timestamp}.h5")

    if not os.path.exists(base_model_path):
        print("Model de baza nu exista!")
        return False

    if not os.path.exists(app_user_images):
        print("Directorul utilizatori nu exista!")
        return False

    people_dict = get_people_dict(app_user_images)
    if len(people_dict) == 0:
        print("Nu s-au gasit utilizatori!")
        return False

    total_images = sum(len(images) for images in people_dict.values())

    negative_images = []
    if os.path.exists(source_negatives):
        negative_images = get_negative_list(source_negatives)

    all_pairs = smart_data_strategy(people_dict, negative_images)
    if len(all_pairs) < 40:
        print("Dataset prea mic!")
        return False

    params = configure_training_params(len(people_dict), total_images, len(all_pairs))

    val_size = int(len(all_pairs) * params['validation_split'])
    random.shuffle(all_pairs)
    train_pairs = all_pairs[val_size:]
    val_pairs = all_pairs[:val_size]

    train_dataset = (make_tf_dataset(train_pairs)
                     .shuffle(buffer_size=len(train_pairs))
                     .map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(params['batch_size'])
                     .prefetch(tf.data.AUTOTUNE))

    val_dataset = (make_tf_dataset(val_pairs)
                   .map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(params['batch_size'])
                   .prefetch(tf.data.AUTOTUNE))

    try:
        model = keras.models.load_model(base_model_path, compile=False)
    except:
        try:
            model = make_siamese_model()
            model.load_weights(base_model_path)
        except:
            print("Eroare incarcare model!")
            return False

    model = optimize_model(model)


    optimizer = keras.optimizers.Adam(
        learning_rate=params['learning_rate'],
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8  # Epsilon mai mic pentru fine-tuning
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    callbacks = create_callbacks(new_model_path, params)

    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=params['epochs'],
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        print(f"Eroare antrenare: {e}")
        return False

    best_threshold, final_metrics = optimize_threshold(model, val_dataset)

    try:
        model.save(new_model_path, save_format='h5', include_optimizer=False)

        test_model = keras.models.load_model(new_model_path, compile=False)
        del test_model

        print(f"Model salvat și verificat cu succes: {new_model_path}")

        # Salveaza modelul cu numele standard pentru aplicatie cu setari speciale
        face_model_path = os.path.join(output_dir, "face_model.h5")
        model.save(face_model_path, save_format='h5', include_optimizer=False, save_traces=False)
        print(f"Model salvat ca face_model.h5: {face_model_path}")

        # Salveaza și weights separat ca backup
        weights_path = new_model_path.replace('.h5', '_weights.h5')
        model.save_weights(weights_path)
        print(f"Weights backup salvate: {weights_path}")

    except Exception as e:
        print(f"Eroare salvare: {e}")

        try:
            print("incerc salvare prin recrearea arhitecturii...")

            weights = model.get_weights()

            new_model = make_siamese_model()
            new_model.set_weights(weights)

            new_model.save(new_model_path, save_format='h5', include_optimizer=False)
            face_model_path = os.path.join(output_dir, "face_model.h5")
            new_model.save(face_model_path, save_format='h5', include_optimizer=False)

            print(f"Salvare prin recreare reușita: {face_model_path}")

        except Exception as e2:
            print(f"Și salvarea prin recreare a eșuat: {e2}")
            return False

    # Config
    config_file = os.path.join(output_dir, f"config_{timestamp}.py")
    face_model_path = os.path.join(output_dir, "face_model.h5")

    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(f"MODEL_ABS_PATH = r'{new_model_path}'\n")
        f.write(f"FACE_MODEL_PATH = r'{face_model_path}'\n")
        f.write(f"THRESHOLD = {final_metrics['threshold']:.3f}\n")
        f.write(f"EXPECTED_ACCURACY = {final_metrics['accuracy']:.4f}\n")
        f.write(f"EXPECTED_F1_SCORE = {final_metrics['f1']:.4f}\n")
        f.write(f"EXPECTED_AUC = {final_metrics['auc']:.4f}\n")

    print(f"Threshold: {final_metrics['threshold']:.3f}")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"F1-Score: {final_metrics['f1']:.4f}")
    print(f"AUC: {final_metrics['auc']:.4f}")
    print(f"Config salvat: {config_file}")

    return True


if __name__ == "__main__":
    main()