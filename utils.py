import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from config import DEVICE

detector = MTCNN(
    image_size=160,
    margin=40,
    keep_all=True,
    device=DEVICE
)

encoder = InceptionResnetV1(
    pretrained='vggface2'
).eval().to(DEVICE)


def get_embedding(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    face = detector(img_pil)

    if face is None:
        return None

    if face.ndim == 4:
        face = face[0]

    face = face.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = encoder(face)

    emb = emb.cpu().numpy()[0]
    emb = emb / np.linalg.norm(emb)

    return emb