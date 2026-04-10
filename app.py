from flask import Flask, request, jsonify
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

from model import process_attendance
from config import MATCH_THRESHOLD

app = Flask(__name__)

# =========================
# 🔥 GLOBAL VARIABLES
# =========================
student_db = {}
student_names = {}

# =========================
# 🔥 FIREBASE INIT
# =========================
import json

firebase_key = os.environ.get("FIREBASE_KEY")

if not firebase_key:
    raise Exception("FIREBASE_KEY not set")

firebase_config = json.loads(firebase_key)

cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred)
db = firestore.client()

# =========================
# 👨‍🎓 LOAD STUDENTS (FAST)
# =========================
def load_students():
    global student_db, student_names

    print("Loading students (NO DOWNLOAD)...")

    students = db.collection("students").stream()

    for student in students:
        sid = student.id
        data = student.to_dict()

        student_names[sid] = data.get("name", sid)

        faces = db.collection("students").document(sid).collection("faces").stream()

        embeddings = []

        for face in faces:
            face_data = face.to_dict()

            embedding = face_data.get("embedding")

            if embedding:
                embeddings.append(np.array(embedding))

        if len(embeddings) > 0:
            student_db[sid] = embeddings

    print("Students loaded:", len(student_db))


# =========================
# 🏠 HOME
# =========================
@app.route("/")
def home():
    return "AI Attendance Backend Running 🚀"


# =========================
# 🎯 ATTENDANCE API
# =========================
@app.route("/process_attendance", methods=["POST"])
def process():
    try:
        # ✅ LOAD ONLY ONCE
        if len(student_db) == 0:
            load_students()

        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data"}), 400

        image_urls = data.get("image_urls", [])
        date = data.get("date")

        if not image_urls or not date:
            return jsonify({"error": "Missing data"}), 400

        print("Processing attendance...")

        present_ids = set()
        total_faces_all = 0
        unknown_total = 0

        # =========================
        # 🔍 PROCESS IMAGES
        # =========================
        for url in image_urls:
            # 🔥 HERE still need image (input image only)
            import requests
            import cv2

            response = requests.get(url, timeout=5)
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                continue

            _, detected_ids, unknown_count, total_faces = process_attendance(
                img,
                student_db,
                MATCH_THRESHOLD
            )

            total_faces_all += total_faces
            unknown_total += unknown_count

            for sid in detected_ids:
                present_ids.add(sid)

        present_ids = list(present_ids)

        total_students = len(student_db)
        present_count = len(present_ids)
        absent_count = total_students - present_count

        # =========================
        # 💾 SAVE TO FIREBASE
        # =========================
        ref = db.collection("attendance").document(date).collection("students")

        for sid in student_db.keys():
            status = "Present" if sid in present_ids else "Absent"
            name = student_names.get(sid, sid)

            ref.document(sid).set({
                "name": name,
                "status": status
            })

            db.collection("students").document(sid)\
                .collection("history").document(date).set({
                    "date": date,
                    "status": status
                })

        return jsonify({
            "present_students": present_ids,
            "present_count": present_count,
            "absent_count": absent_count,
            "total_students_in_db": total_students,
            "detected_faces": total_faces_all,
            "unknown_count": unknown_total,
            "attendance_percentage": round((present_count / total_students) * 100, 2)
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# =========================
# 🚀 RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)