import os
import random
import tensorflow as tf

def get_people_dict(people_path):
    people_dict = {}
    for person in os.listdir(people_path):
        person_folder = os.path.join(people_path, person)
        if os.path.isdir(person_folder):
            images = [os.path.join(person_folder, f) for f in os.listdir(person_folder)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) > 1:
                people_dict[person] = images
                print(f"Persoana '{person}': {len(images)} imagini")
    return people_dict

def get_negative_list(neg_path):
    negatives = [os.path.join(neg_path, f) for f in os.listdir(neg_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Total imagini negative: {len(negatives)}")
    return negatives

def generate_pairs(people_dict, negatives, pairs_per_person=90, neg_pairs_between_people=30):
    positive_pairs = []
    negative_pairs = []

    for person, images in people_dict.items():
        for _ in range(pairs_per_person):
            if len(images) >= 2:
                img1, img2 = random.sample(images, 2)
                positive_pairs.append((img1, img2, 1))

    for person, images in people_dict.items():
        for _ in range(pairs_per_person):
            img1 = random.choice(images)
            img_neg = random.choice(negatives)
            negative_pairs.append((img1, img_neg, 0))

    persons = list(people_dict.keys())
    if len(persons) >= 2:
        for _ in range(neg_pairs_between_people * len(persons)):
            p1, p2 = random.sample(persons, 2)
            img1 = random.choice(people_dict[p1])
            img2 = random.choice(people_dict[p2])
            negative_pairs.append((img1, img2, 0))
    else:
        print(f"[EROARE] Nu exista cel puțin 2 persoane pentru generarea perechilor negative intre persoane: {persons}")

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    print(f"Perechi pozitive: {len(positive_pairs)}")
    print(f"Perechi negative: {len(negative_pairs)}")
    print(f"Total perechi: {len(all_pairs)}")
    return all_pairs

def make_tf_dataset(pairs):
    path1 = [p[0] for p in pairs]
    path2 = [p[1] for p in pairs]
    labels = [p[2] for p in pairs]
    dataset = tf.data.Dataset.from_tensor_slices((path1, path2, labels))
    return dataset
