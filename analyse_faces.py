#!/usr/bin/env python3
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

SOURCE_IMG_PATH = 'data/src.jpg'
TARGET_IMG_PATH = 'data/dest.jpg'


def get_face_analyser(model_name: str, model_path: str, det_size=(320, 320)):
    face_analyser = FaceAnalysis(
        name=model_name,
        root=model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_one_face(face_analyser, frame:np.ndarray):
    try:
        face = face_analyser.get(frame)
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(face_analyser, frame:np.ndarray):
    """
    get faces from left to right by order
    """
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def count_faces():
    face_analyser = get_face_analyser('buffalo_l', './')

    source_img = Image.open(SOURCE_IMG_PATH)
    source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    source_faces = get_many_faces(face_analyser, source_img)
    num_source_faces = len(source_faces)

    target_img = Image.open(TARGET_IMG_PATH)
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_faces = get_many_faces(face_analyser, target_img)
    num_target_faces = len(target_faces)

    return num_source_faces, num_target_faces


if __name__ == '__main__':
    num_source_faces, num_target_faces = count_faces()

    print(f'Source faces: {num_source_faces}')
    print(f'Target faces: {num_target_faces}')
