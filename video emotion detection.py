import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse
from tqdm import tqdm

# ----------------------
# Dataset Loader
# ----------------------
class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))

        for idx, label in enumerate(self.classes):
            class_folder = os.path.join(root_dir, label)
            for file in os.listdir(class_folder):
                self.samples.append((os.path.join(class_folder, file), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label

# ----------------------
# CNN + RNN Model
# ----------------------
class CNNRNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNRNNModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.rnn = nn.LSTM(128 * 8 * 8, 256, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        cnn_out = self.cnn(x.view(batch_size * seq_len, c, h, w))
        cnn_out = cnn_out.view(batch_size, seq_len, -1)
        rnn_out, _ = self.rnn(cnn_out)
        return self.fc(rnn_out[:, -1, :])

# ----------------------
# Training Function
# ----------------------
# ... [rest of your code unchanged] ...

def train_model():
    dataset_dir = os.path.join(os.getcwd(), "fer2013", "train")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset not found at {dataset_dir}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = EmotionDataset(dataset_dir, transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNRNNModel(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1):  
        total_loss = 0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.unsqueeze(1).to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "emotion_model.pth")
    print("✅ Training completed & model saved!")

# ... [rest of your code unchanged] ...


# ----------------------
# Inference Function
# ----------------------
def infer_video(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = os.path.join(os.getcwd(), "fer2013", "train")
    dummy_dataset = EmotionDataset(dataset_dir, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ]))
    model = CNNRNNModel(num_classes=len(dummy_dataset.classes)).to(device)
    model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = dummy_dataset.transform(frame_rgb)
        frames.append(frame_tensor)

    cap.release()
    if not frames:
        print("❌ No frames extracted from video.")
        return

    frames_tensor = torch.stack(frames).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(frames_tensor)
        predicted = torch.argmax(output, dim=1).item()
        print(f"Predicted Emotion: {dummy_dataset.classes[predicted]}")

# ----------------------
# Main CLI
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--infer", type=str, help="Infer emotion from video")
    args = parser.parse_args()

    if args.train:
            train_model()
    elif args.infer:
        infer_video(args.infer)
    else:
        print("No action specified. Use --train or --infer <video_path>")
