import random
import tensorflow as tf
from utils.preprocessing import preprocess

def verify_pair(model, img1_path, img2_path, threshold=0.5):
    img1 = preprocess(img1_path)
    img2 = preprocess(img2_path)
    img1 = tf.expand_dims(img1, 0)
    img2 = tf.expand_dims(img2, 0)
    prediction = model([img1, img2])
    confidence = prediction[0][0].numpy()
    is_same = confidence > threshold
    return is_same, confidence

def test_model_samples(model, people_dict, negatives, num_tests=3):
    print("\nTestare Model")
    for _ in range(num_tests):
        person = random.choice(list(people_dict.keys()))
        if len(people_dict[person]) < 2:
            continue
        img1, img2 = random.sample(people_dict[person], 2)
        is_same, confidence = verify_pair(model, img1, img2)
        print(f"[+][{person}] -> {is_same} (Conf: {confidence:.4f})")

        p1, p2 = random.sample(list(people_dict.keys()), 2)
        img1 = random.choice(people_dict[p1])
        img2 = random.choice(people_dict[p2])
        is_same, confidence = verify_pair(model, img1, img2)
        print(f"[-] {p1} vs {p2} -> {is_same} (Conf: {confidence:.4f})")

        p = random.choice(list(people_dict.keys()))
        img1 = random.choice(people_dict[p])
        img2 = random.choice(negatives)
        is_same, confidence = verify_pair(model, img1, img2)
        print(f"[-] {p} vs [neg] -> {is_same} (Conf: {confidence:.4f})")
