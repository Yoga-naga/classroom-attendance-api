import cv2
import numpy as np
import torch
from PIL import Image
from utils import detector, encoder
from config import DEVICE, GAP_THRESHOLD


def process_attendance(group_img, database, threshold):

    img_rgb = cv2.cvtColor(group_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    boxes, _ = detector.detect(img_pil)
    faces = detector(img_pil)

    present_ids = []
    unknown_count = 0
    total_faces = 0

    if faces is None or boxes is None:
        print("No faces detected")
        return group_img, [], 0, 0

    faces = faces.to(DEVICE)
    total_faces = len(faces)

    with torch.no_grad():
        embeddings = encoder(faces).cpu().numpy()

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    for i, g_emb in enumerate(embeddings):

        best_id = None
        best_sim = -1
        second_best = -1

        # 🔥 MULTI-EMBEDDING MATCH
        for sid, emb_list in database.items():

            if len(emb_list) == 0:
                continue

            for ref_emb in emb_list:

                sim = np.dot(g_emb, ref_emb)

                if sim > best_sim:
                    second_best = best_sim
                    best_sim = sim
                    best_id = sid
                elif sim > second_best:
                    second_best = sim

        print(f"Best: {best_id} ({best_sim}) | Second: {second_best}")

        # ✅ FINAL DECISION
        if best_sim >= threshold and (best_sim - second_best) >= GAP_THRESHOLD:
            final_id = best_id
            present_ids.append(best_id)
            color = (0, 255, 0)
        else:
            final_id = "Unknown"
            unknown_count += 1
            color = (0, 0, 255)

        box = boxes[i].astype(int)

        cv2.rectangle(group_img, (box[0], box[1]), (box[2], box[3]), color, 2)

        cv2.putText(
            group_img,
            f"{final_id} ({best_sim:.2f})",
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return group_img, present_ids, unknown_count, total_faces