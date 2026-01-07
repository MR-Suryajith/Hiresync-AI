import cv2
import mediapipe as mp
import numpy as np
import os

print("🚀 HireSync Phase 2 Gaze ✅")

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

while cap.isOpened():
    ret, frame = cap
