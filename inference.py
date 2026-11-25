from pathlib import Path
import numpy as np
import cv2
import joblib
import torch
from torch import nn
from torchvision import models, transforms
from skimage.feature import hog

class MangoCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))

def build_googlenet(num_classes, dropout=0.4):
    model = models.googlenet(weights=None, aux_logits=False,
                             transform_input=True, init_weights=False)
    model.fc = nn.Sequential(nn.Dropout(p=dropout),
                             nn.Linear(model.fc.in_features, num_classes))
    return model

FEAT_SIZE, HOG_CELL_SIZE, HOG_NBINS = (96, 96), (16, 16), 9

def compute_lbp_histogram(gray_u8, n_bins=256):
    center = gray_u8[1:-1, 1:-1]
    neighbors = [gray_u8[:-2, :-2], gray_u8[:-2, 1:-1], gray_u8[:-2, 2:], gray_u8[1:-1, 2:],
                 gray_u8[2:, 2:], gray_u8[2:, 1:-1], gray_u8[2:, :-2], gray_u8[1:-1, :-2]]
    lbp_code = np.zeros_like(center, dtype=np.uint8)
    for bit_idx, nb in enumerate(neighbors):
        lbp_code |= ((nb >= center).astype(np.uint8) << bit_idx)
    hist_lbp, _ = np.histogram(lbp_code, bins=n_bins, range=(0, 256))
    hist_lbp = hist_lbp.astype(np.float32)
    return hist_lbp / (hist_lbp.sum() + 1e-7)

def extract_features(pil_image):
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, FEAT_SIZE, interpolation=cv2.INTER_AREA)
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hists = []
    for ch, rng in ((0, [0, 180]), (1, [0, 256]), (2, [0, 256])):
        h = cv2.calcHist([img_hsv], [ch], None, [16], rng).flatten()
        hists.append((h / (h.sum() + 1e-7)).astype(np.float32))
    gray_u8 = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray_u8, orientations=HOG_NBINS, pixels_per_cell=HOG_CELL_SIZE,
                   cells_per_block=(1, 1), visualize=False).astype(np.float32)
    lbp_feat = compute_lbp_histogram(gray_u8, 256)
    return np.concatenate(hists + [hog_feat, lbp_feat]).astype(np.float32)

class Classifier:
    def __init__(self, folder):
        folder = Path(folder)
        for name in ("cnn.pth", "googlenet.pth", "svm.joblib"):
            if not (folder / name).is_file():
                raise FileNotFoundError(f"Missing trained model: {folder / name}. See README.")
        c = torch.load(folder / "cnn.pth", map_location="cpu", weights_only=True)
        g = torch.load(folder / "googlenet.pth", map_location="cpu", weights_only=True)
        # Only load joblib artifacts from trusted sources.
        s = joblib.load(folder / "svm.joblib")
        self.classes = list(c["class_names"])
        if self.classes != list(g["class_names"]) or self.classes != list(s["class_names"]):
            raise ValueError("Model class orders do not match")
        self.cnn = MangoCNN(len(self.classes))
        self.cnn.load_state_dict(c["model_state"])
        self.cnn.eval()
        self.gnet = build_googlenet(len(self.classes))
        self.gnet.load_state_dict(g["model_state"])
        self.gnet.eval()
        self.svm = s["model"]
        self.transforms = [transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize(k["mean"], k["std"])
        ]) for k in (c, g)]

    def predict(self, image):
        image = image.convert("RGB")
        feat = extract_features(image).reshape(1, -1)
        svm_probs = self.svm.predict_proba(feat)[0]
        result = {"SVM": {self.classes[int(i)]: float(p) for i,p in zip(self.svm.classes_, svm_probs)}}
        with torch.inference_mode():
            for name, model, tf in zip(("CNN", "GoogLeNet"), (self.cnn, self.gnet), self.transforms):
                probs = model(tf(image).unsqueeze(0)).softmax(1)[0].tolist()
                result[name] = dict(zip(self.classes, probs))
        return result
