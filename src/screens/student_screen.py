import streamlit as st
import numpy as np
from PIL import Image
import time

from src.ui.base_layout import style_background_dashboard, style_base_layout
from src.components.header import header_dashboard
from src.components.subject_card import subject_card
from src.components.dialog_enroll import enroll_dialog
from src.pipelines.face_pipeline import predict_attendance, get_face_embeddings, train_classifier
from src.pipelines.voice_pipeline import get_voice_embedding
from src.database.db import (
    get_all_students,
    create_student,
    get_student_subjects,
    get_student_attendance,
    unenroll_student_to_subject
)


def student_dashboard():
    """Display the student dashboard with enrolled subjects."""
    
    # Get student info from session
    student_data = st.session_state.student_data
    student_id = student_data['student_id']
    
    # ====================
    # HEADER SECTION
    # ====================
    col_header, col_welcome = st.columns(2, vertical_alignment='center', gap='xxlarge')
    
    with col_header:
        header_dashboard()
    
    with col_welcome:
        st.subheader(f"Welcome, {student_data['name']}")
        if st.button("Logout", type='secondary', key='logout_btn', shortcut="control+backspace"):
            st.session_state['is_logged_in'] = False
            del st.session_state.student_data
            st.rerun()

    st.space()
    
    # ====================
    # SUBJECTS SECTION HEADER
    # ====================
    col_title, col_enroll_btn = st.columns(2)
    
    with col_title:
        st.header('Your Enrolled Subjects')
    
    with col_enroll_btn:
        if st.button('Enroll in Subject', type='primary', width='stretch'):
            enroll_dialog()

    st.divider()

    # ====================
    # LOAD DATA
    # ====================
    with st.spinner('Loading your enrolled subjects...'):
        subjects = get_student_subjects(student_id)
        attendance_logs = get_student_attendance(student_id)

    # ====================
    # CALCULATE ATTENDANCE STATS
    # ====================
    stats_map = {}
    
    for log in attendance_logs:
        subject_id = log['subject_id']
        
        # Initialize stats if not exists
        if subject_id not in stats_map:
            stats_map[subject_id] = {"total": 0, "attended": 0}
        
        # Count total attendance records
        stats_map[subject_id]['total'] += 1
        
        # Count attended records
        if log.get('is_present'):
            stats_map[subject_id]['attended'] += 1

    # ====================
    # DISPLAY SUBJECT CARDS
    # ====================
    cols = st.columns(2)
    
    for index, subject_node in enumerate(subjects):
        subject = subject_node['subjects']
        subject_id = subject['subject_id']
        
        # Get stats for this subject
        subject_stats = stats_map.get(
            subject_id,
            {"total": 0, "attended": 0}
        )
        
        # Define the unenroll button callback
        def handle_unenroll(sid):
            """Handle unenroll button click."""
            if st.button(
                "Unenroll from this course",
                type='tertiary',
                width='stretch',
                icon=':material/delete_forever:',
                key=f"unenroll_{sid}"  # Unique key for each subject
            ):
                unenroll_student_to_subject(student_id, sid)
                st.toast('Unenrolled successfully!')
                st.rerun()
        
        # Display subject card in the appropriate column
        with cols[index % 2]:
            subject_card(
                name=subject['name'],
                code=subject['subject_code'],
                section=subject['section'],
                stats=[
                    ('📅', 'Total', subject_stats['total']),
                    ('✅', 'Attended', subject_stats['attended']),
                ],
                footer_callback=handle_unenroll,
                footer_callback_data=subject_id  # Pass subject ID to callback
            )


def student_screen():
    """Main student login and dashboard screen."""
    
    # Apply styling
    style_background_dashboard()
    style_base_layout()

    # ====================
    # IF ALREADY LOGGED IN - SHOW DASHBOARD
    # ====================
    if "student_data" in st.session_state:
        student_dashboard()
        return
    
    # ====================
    # HEADER FOR LOGIN PAGE
    # ====================
    col_header, col_back_btn = st.columns(2, vertical_alignment='center', gap='xxlarge')
    
    with col_header:
        header_dashboard()
    
    with col_back_btn:
        if st.button("Go back to Home", type='secondary', key='back_to_home_btn', shortcut="control+backspace"):
            st.session_state['login_type'] = None
            st.rerun()

    # ====================
    # LOGIN SECTION
    # ====================
    st.header('Login using FaceID', text_alignment='center')
    st.space()
    st.space()
    
    show_registration = False
    photo_source = st.camera_input("Position your face in the center")

    if photo_source:
        img = np.array(Image.open(photo_source))

        with st.spinner('AI is scanning...'):
            detected, all_ids, num_faces = predict_attendance(img)

            # Check if exactly one face is detected
            if num_faces == 0:
                st.warning('❌ Face not found!')
            elif num_faces > 1:
                st.warning('❌ Multiple faces found')
            else:
                if detected:
                    # Face recognized - find student in database
                    detected_student_id = list(detected.keys())[0]
                    all_students = get_all_students()
                    student = next(
                        (s for s in all_students if s['student_id'] == detected_student_id),
                        None
                    )

                    if student:
                        # Student found - log them in
                        st.session_state.is_logged_in = True
                        st.session_state.user_role = 'student'
                        st.session_state.student_data = student
                        st.toast(f"✅ Welcome Back {student['name']}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        # Student not found - show registration form
                        st.warning('Student record not found. Please register below.')
                        show_registration = True
                else:
                    # Face not recognized
                    st.info('😊 Face not recognized! You might be a new student.')
                    show_registration = True

    # ====================
    # REGISTRATION SECTION (if needed)
    # ====================
    if show_registration:
        with st.container(border=True):
            st.header('Register new Profile')
            
            # Get student name
            new_name = st.text_input(
                "Enter your name",
                placeholder='E.g. Hamza Rizvi'
            )

            # Optional voice enrollment
            st.subheader('Optional: Voice Enrollment')
            st.info("Enroll your voice for voice-based attendance")

            audio_data = None
            try:
                audio_data = st.audio_input(
                    'Record a short phrase: "I am present" or "My name is [Your Name]"'
                )
            except Exception as e:
                st.error(f'Audio recording failed: {e}')

            # Create account button
            if st.button('Create Account', type='primary'):
                if new_name:
                    with st.spinner('Creating your profile...'):
                        img = np.array(Image.open(photo_source))
                        
                        # Extract facial features
                        face_encodings = get_face_embeddings(img)
                        
                        if face_encodings:
                            face_embedding = face_encodings[0].tolist()

                            # Extract voice features if provided
                            voice_embedding = None
                            if audio_data:
                                voice_embedding = get_voice_embedding(audio_data.read())

                            # Create student in database
                            response_data = create_student(
                                new_name,
                                face_embedding=face_embedding,
                                voice_embedding=voice_embedding
                            )

                            if response_data:
                                # Train the face classifier with new student
                                train_classifier()
                                
                                # Log them in
                                st.session_state.is_logged_in = True
                                st.session_state.user_role = 'student'
                                st.session_state.student_data = response_data[0]
                                st.toast(f'✅ Profile Created! Hi {new_name}!')
                                time.sleep(1)
                                st.rerun()
                        else:
                            st.error('❌ Could not capture your facial features. Please try again.')
                else:
                    st.warning('⚠️ Please enter your name!')