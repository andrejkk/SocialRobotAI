import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models.video import ViViT
import numpy as np
import cv2
import mediapipe as mp

from extract_video import ProcessFrames
from datasets import load_dataset


# Custom Dataset for event-labeled video frames with pose and face labels
class EventLabeledVideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels  # Dict with 'pose' and 'face' keys
        self.transform = transform
        self.mp_pose = mp.solutions.pose.Pose()
        self.mp_face = mp.solutions.face_detection.FaceDetection()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # pose_landmarks = []
        # face_boxes = []
        # for frame in frames:
        #     results_pose = self.mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #     if results_pose.pose_landmarks:
        #         pose_landmarks.append(
        #             [[lm.x, lm.y, lm.z] for lm in results_pose.pose_landmarks.landmark]
        #         )
        #     else:
        #         pose_landmarks.append([])

        #     results_face = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #     if results_face.detections:
        #         face_boxes.append(
        #             [results_face.detections[0].location_data.relative_bounding_box]
        #         )
        #     else:
        #         face_boxes.append([])

        # pretvoriti v tenzorje
        video_tensor = torch.stack([self.transform(frame) for frame in frames])
        pose_label = torch.tensor(self.labels[idx]["pose"], dtype=torch.float32)
        face_label = torch.tensor(self.labels[idx]["face"], dtype=torch.float32)

        return video_tensor, pose_label, face_label


# ViViT Model z batch norm
class ViViTWithBN(nn.Module):
    def __init__(self, num_classes_pose, num_classes_face):
        super(ViViTWithBN, self).__init__()
        self.vivit = ViViT(weights=None, num_classes=512)
        self.bn = nn.BatchNorm3d(3)  # Batch norm for video frames (C, T, H, W)
        self.pose_head = nn.Linear(512, num_classes_pose)
        self.face_head = nn.Linear(512, num_classes_face)

    def forward(self, x):
        x = self.bn(x)
        features = self.vivit(x)
        pose_out = self.pose_head(features)
        face_out = self.face_head(features)
        return pose_out, face_out


def train_and_evaluate(model, train_loader, val_loader, num_epochs=10):
    criterion = nn.MSELoss()  # za opravilo regressiona za landmark detection
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        for videos, pose_labels, face_labels in train_loader:
            optimizer.zero_grad()
            pose_pred, face_pred = model(videos)
            loss = criterion(pose_pred, pose_labels) + criterion(face_pred, face_labels)
            loss.backward()
            optimizer.step()

    model.eval()
    pose_acc = 0
    face_acc = 0
    with torch.no_grad():
        for videos, pose_labels, face_labels in val_loader:
            pose_pred, face_pred = model(videos)
            # Simple accuracy metric (e.g., mean absolute error < threshold)

            pose_acc += (torch.abs(pose_pred - pose_labels) < 0.1).float().mean()
            face_acc += (torch.abs(face_pred - face_labels) < 0.1).float().mean()
    pose_acc /= len(val_loader)
    face_acc /= len(val_loader)
    print(f"Pose Tracking Accuracy: {pose_acc:.4f}")
    print(f"Face Tracking Accuracy: {face_acc:.4f}")


# Main
if __name__ == "__main__":
    # video_paths = ["video1.mp4", "video2.mp4"]
    # labels = [
    #     {"pose": np.random.rand(33, 3), "face": np.random.rand(1, 4)},
    #     {"pose": np.random.rand(33, 3), "face": np.random.rand(1, 4)},
    # ]

    video_files = []
    for root, dirs, files in os.walk("Data"):
        print("Searching for files in: ", root)
        for file in files:
            if os.path.isfile(os.path.join(root, file)) and file.lower().endswith(
                ".webm"
            ):
                # print("Found videofile: ", file)
                video_files.append(os.path.join(root, file))

    ############ PROCESS FRAMES FOR EACH VIDEO ############
    csvid = 0
    for video in video_files:
        pf = ProcessFrames(video)
        df = pf.process_frames()
        df.to_csv(f"timeseries_{csvid}.csv", index=False)
        print(f"Saved {len(df)} frames to timeseries.csv from video {video}")
        csvid += 1

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("jili5044/hmdb51")
    labels = {
        "pose": "handsup",
        "attention": "positive",
    }  # Replace with actual label keys

    dataset = EventLabeledVideoDataset(video_paths, labels, transform)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = ViViTWithBN(num_classes_pose=33 * 3, num_classes_face=4)
    train_and_evaluate(model, train_loader, val_loader)
