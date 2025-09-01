from mtcnn.mtcnn import MTCNN
import cv2
import os

img_size = 105

SOURCE_DIR = "./data/celebrity_faces_dataset"
DEST_DIR = "./data/positives_anchors"
SOURCE_DIR_NEG = "./data/non-celebrity_faces_dataset"
DEST_DIR_NEG = "./data/negatives"

def crop_faces():
    os.makedirs(DEST_DIR, exist_ok=True)
    os.makedirs(DEST_DIR_NEG, exist_ok=True)
    detector = MTCNN()

    for person in os.listdir(SOURCE_DIR):
        person_src_folder = os.path.join(SOURCE_DIR, person)
        person_dst_folder = os.path.join(DEST_DIR, person)
        os.makedirs(person_dst_folder, exist_ok=True)

        for img_name in os.listdir(person_src_folder):
            src_path = os.path.join(person_src_folder, img_name)
            dst_path = os.path.join(person_dst_folder, img_name)

            img = cv2.imread(src_path)
            if img is None:
                print(f"[WARNING] Nu se paote citi {src_path}")
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(rgb_img)
            if len(results) == 0:
                print(f"[WARNING] Nu am gasit fata în {src_path}")
                continue

            x, y, w, h = results[0]['box']
            x, y = max(x, 0), max(y, 0)
            face = rgb_img[y:y+h, x:x+w]
            if face.size == 0:
                print(f"[WARNING] Crop invalid pentru {src_path}")
                continue

            face = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

    for img_name in os.listdir(SOURCE_DIR_NEG):
        src_path = os.path.join(SOURCE_DIR_NEG, img_name)
        dst_path = os.path.join(DEST_DIR_NEG, img_name)

        img = cv2.imread(src_path)
        if img is None:
            print(f"[WARNING] Nu pot citi {src_path}")
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_img)
        if len(results) == 0:
            print(f"[WARNING] Nu am gasit fata în {src_path}")
            continue

        x, y, w, h = results[0]['box']
        x, y = max(x, 0), max(y, 0)
        face = rgb_img[y:y+h, x:x+w]
        if face.size == 0:
            print(f"[WARNING] Crop invalid pentru {src_path}")
            continue

        face = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

def crop_reference_face(reference_image_path, detector, img_size=105):
    if not os.path.exists(reference_image_path):
        print(f"[ERROR] Imaginea de referinta nu exista: {reference_image_path}")
        return None

    img = cv2.imread(reference_image_path)
    if img is None:
        print(f"[ERROR] Nu pot citi imaginea: {reference_image_path}")
        return None

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_img)
    if len(results) == 0:
        print(f"[WARNING] Nu am gasit nicio fata în imaginea de referinta.")
        return None

    x, y, w, h = results[0]['box']
    x, y = max(x, 0), max(y, 0)
    face = rgb_img[y:y + h, x:x + w]
    if face.size == 0:
        print(f"[WARNING] Crop invalid în imaginea de referinta.")
        return None

    face = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)
    save_dir = os.path.dirname(reference_image_path)
    save_path = os.path.join(save_dir, "reference_crop.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Fata a fost salvata ca {save_path}")
    return save_path


