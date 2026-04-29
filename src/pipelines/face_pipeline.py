import dlib
import face_recognition_models
import numpy as np 
from sklearn.svm import SVC
import streamlit as st 

from src.database.db import get_all_students 

@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector()
    sp= dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )
    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec

def get_face_embedding(image_np):
    detector , sp , facerec = load_dlib_models()
    faces = detector(image_np, 1)
    encoding = []
    for face in faces:
        shape= sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1)
        encoding.append(np.array(face_descriptor))
        return encoding
    
@st.cache_resource
def get_train_models():
    X = []
    y = []

    students_db = get_all_students()
    if not students_db:
        return None

    for student in students_db:
        embedding =  student.get("face_embedding") 
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get('student_id'))
    return X, y 

    if len(X) == 0:
        return 0
    
    clf = SVC(kernel='linear',probability=True, class_weight='balanced')


    try:
        clif.fit(X, y)
    except ValueError:
        pass     
    return{"clf":clf , "X":x, "y":y}

def train_classifier():
    st.cache_resource.clear()
    model_data = get_train_models()
    return bool(model_data)


def predict_attendance(class_image_np):
    encodings = get_face_embedding(class_image_np)

    detected_students = {}

    model_data = get_train_models()

    if not model_data:
        return detected_students, [], len(encodings)
    
    clf = model_data['clf']
    X_train = model_data['X']
    y_train = model_data['y']


    all_students = sorted(list(set(y_train)))

    for encoding in encodings:
        if len(all_students) >= 2:
            predicted_id = int(clf.predict([encoding])[0])
        else :
            predicted_id = int(all_students[0])    

        student_embedding = X_train[y_train.index(predicted_id)]    

        best_match_score = np.linalg.norm(student_embedding - encoding)
        resemblance_threshold = 0.6

        if best_match_score <= resemblance_threshold:
            detected_students[predicted_id] = True

    return detected_students, all_students, len(encodings)       
