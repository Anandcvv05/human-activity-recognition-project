import os
import cv2
import numpy as np

IMG_SIZE = 70
MAX_FRAMES = 25


def extract_frames(video_path, max_frames=MAX_FRAMES, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        return np.array(frames)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return np.array(frames)

    step = max(1, total_frames // max_frames)
    current_frame = 0

    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype("float32") / 255.0
        frames.append(frame)

        current_frame += step
        if current_frame >= total_frames:
            break

    cap.release()

    while len(frames) < max_frames:
        frames.append(np.zeros((img_size, img_size, 3), dtype=np.float32))

    return np.array(frames)


def load_dataset(dataset_path, class_names):
    X = []
    y = []

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_path, class_name)

        if not os.path.exists(class_folder):
            print(f"Warning: Folder not found -> {class_folder}")
            continue

        for file_name in os.listdir(class_folder):
            if file_name.lower().endswith((".avi", ".mp4", ".mov")):
                video_path = os.path.join(class_folder, file_name)
                frames = extract_frames(video_path)

                if frames.shape[0] == MAX_FRAMES:
                    X.append(frames)
                    y.append(label)

    return np.array(X), np.array(y)