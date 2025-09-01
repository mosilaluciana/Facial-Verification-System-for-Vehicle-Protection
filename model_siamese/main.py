import os
import cv2
import random

from sklearn.metrics import accuracy_score, f1_score
from scripts.data_loader import get_people_dict, get_negative_list, generate_pairs, make_tf_dataset
from scripts.training import train_model, plot_training_history
from scripts.evaluation import test_model_samples
from models.siamese_model import make_siamese_model
from utils.preprocessing import preprocess_twin_train, preprocess_twin_val
import numpy as np
import tensorflow as tf

PEOPLE_PATH = os.path.join('data', 'positives_anchors')
NEG_PATH = os.path.join('data', 'negatives')

def verify_image_shapes(people_dict, negatives, expected_size=(105, 105)):
    errors = 0
    for person, paths in people_dict.items():
        for img_path in paths:
            img = cv2.imread(img_path)
            if img is None or img.shape[:2] != expected_size:
                print(f"[EROARE] Imagine invalida: {img_path}")
                errors += 1
    for img_path in negatives:
        img = cv2.imread(img_path)
        if img is None or img.shape[:2] != expected_size:
            print(f"[EROARE] Imagine negativa invalida: {img_path}")
            errors += 1
    if errors > 0:
        print(f"[AVERTISMENT] {errors} imagini nu au dimensiunea {expected_size[0]}x{expected_size[1]}. Verifica crop-ul.")
        exit(1)

def main():
    print("=== Recunoastere Faciala - Siamese Network ===")

    if not os.path.exists(PEOPLE_PATH) or not os.path.exists(NEG_PATH):
        print("[EROARE] Verifica ca directoarele cu date exista si contin imagini preprocesate.")
        return

    print("\n1. Incarcare date...")
    people_dict = get_people_dict(PEOPLE_PATH)
    negatives = get_negative_list(NEG_PATH)
    verify_image_shapes(people_dict, negatives)

    if not people_dict or not negatives:
        print("[EROARE] Nu s-au gasit suficiente imagini pentru antrenare.")
        return

    print("\n2. Split persoane train/val...")
    all_persons = list(people_dict.keys())
    random.shuffle(all_persons)
    train_person_count = int(0.6 * len(all_persons))
    train_persons = set(all_persons[:train_person_count])
    val_persons = set(all_persons[train_person_count:])

    train_people_dict = {k: v for k, v in people_dict.items() if k in train_persons}
    val_people_dict = {k: v for k, v in people_dict.items() if k in val_persons}

    print(f"[INFO] Persoane train: {len(train_people_dict)}, persoane val: {len(val_people_dict)}")

    print("\n3. Generare perechi train...")
    train_pairs = generate_pairs(train_people_dict, negatives)
    print("Generare perechi val...")
    val_pairs = generate_pairs(val_people_dict, negatives)

    print(f"Perechi pozitive: {sum(1 for _, _, l in train_pairs if l == 1)}")
    print(f"Perechi negative: {sum(1 for _, _, l in train_pairs if l == 0)}")
    print(f"Total perechi: {len(train_pairs)}")
    print(f"Val pozitive: {sum(1 for _, _, l in val_pairs if l == 1)}")
    print(f"Val negative: {sum(1 for _, _, l in val_pairs if l == 0)}")
    print(f"Val total: {len(val_pairs)}")

    positive_pairs = [pair for pair in train_pairs if pair[2] == 1]
    negative_pairs = [pair for pair in train_pairs if pair[2] == 0]
    min_len = min(len(positive_pairs), len(negative_pairs))
    positive_pairs = random.sample(positive_pairs, min_len)
    negative_pairs = random.sample(negative_pairs, min_len)
    train_pairs = positive_pairs + negative_pairs
    random.shuffle(train_pairs)
    print(f"[INFO] Dupa balansare: pozitive: {len(positive_pairs)}, negative: {len(negative_pairs)}, total: {len(train_pairs)}")

    print("\n4. Creare dataset...")
    train_dataset = make_tf_dataset(train_pairs).shuffle(5000).map(preprocess_twin_train).batch(8).prefetch(tf.data.AUTOTUNE)
    val_dataset = make_tf_dataset(val_pairs).map(preprocess_twin_val).batch(8).prefetch(tf.data.AUTOTUNE)

    print("\n5. Creare modelvechi...")
    model = make_siamese_model()
    model.summary()

    print("\n6. Antrenare modelvechi...")
    history = train_model(model, train_dataset, val_dataset)
    plot_training_history(history)
    model.save('siamese_finetuned_best.h5')
    print("Model salvat ca 'siamese_finetuned_best.h5'")

    y_true, y_pred = [], []
    for batch in val_dataset:
        x = batch[:2]
        y = batch[2].numpy().reshape(-1)
        pred = model(x, training=False).numpy().reshape(-1)
        y_true.append(y)
        y_pred.append(pred)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Threshold tuning
    best_acc, best_thresh = 0, 0
    for t in np.arange(0.1, 0.9, 0.01):
        acc = accuracy_score(y_true, (y_pred > t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    print(f'Best threshold: {best_thresh:.2f}, best validation accuracy: {best_acc:.3f}')

    print("\n7. Evaluare modelvechi pe exemple (val set)...")
    test_model_samples(model, val_people_dict, negatives)

    print("\n=== Proces complet finalizat! ===")

if __name__ == "__main__":
    main()
