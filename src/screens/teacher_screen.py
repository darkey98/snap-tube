import streamlit as st
from src.ui.base_layout import style_background_dashboard, style_base_layout
from src.components.header import header_dashboard
from src.database.db import check_teacher_exists, create_teacher, teacher_login, get_teacher_subjects , get_attendance_for_teacher
from src.components.dialog_create_subject import create_subject_dialog
from src.components.subject_card import subject_card
from src.components.dialog_share_subject import share_subject_dialog
from src.components.dialog_add_photos import add_photo_dialog
from src.database.config import supabase
from src.pipelines.face_pipeline import predict_attendance
from src.components.dialog_attendance_results import attendance_result_dialog
from src.components.dialog_voice_attendance import voice_attendance_dialog
import numpy as np
from datetime import datetime
import pandas as pd 

def teacher_screen():
    style_background_dashboard()
    style_base_layout()


    if "teacher_data" in st.session_state:
        teacher_dashboard()
    elif 'teacher_login_type' not in st.session_state or st.session_state.teacher_login_type == "login":
        teacher_screen_login()
    elif st.session_state.teacher_login_type == "register":
        teacher_screen_register()   

def teacher_dashboard():
    teacher_data = st.session_state.teacher_data
    c1 , c2  = st.columns(2, vertical_alignment="center", gap="xxlarge")
    with c1:
        header_dashboard()
    with c2:
        st.subheader(f"""Welcome {teacher_data['name']}""")
        if st.button("Logout", type="secondary", key ='loginbackbtn', shortcut="control+backspace"):
            st.session_state['is_logged_in'] = False 
            del st.session_state.teacher_data
            st.rerun()

            st.space()
            st.space()
    st.space()

    if "current_teacher_tab" not in st.session_state:
        st.session_state.current_teacher_tab = "take_attendance"

    tab1, tab2, tab3 = st.columns(3)    
    with tab1:
        type1='primary' if st.session_state.current_teacher_tab == "take_attendance" else 'tertiary'
        if st.button('Take Attendance',width='stretch',type = type1, icon=":material/ar_on_you:" ):
            st.session_state.current_teacher_tab = "take_attendance"
            st.rerun()
    with tab2:
        type2='primary' if st.session_state.current_teacher_tab == "manage_subjects" else 'tertiary'
        if st.button('Manage Subjects',width='stretch',type = type2, icon=":material/book_ribbon:" ):
            st.session_state.current_teacher_tab = "manage_subjects"
            st.rerun()
    with tab3:
        type3='primary' if st.session_state.current_teacher_tab == "attendance_records" else 'tertiary'
        if st.button('Attendance Records',width='stretch',type = type3, icon=":material/cards_stack:" ):
            st.session_state.current_teacher_tab = "attendance_records"
            st.rerun()    
    
    st.divider()

    if st.session_state.current_teacher_tab == "take_attendance":
        teacher_tab_take_attendance()       
    if st.session_state.current_teacher_tab == "manage_subjects":
        teacher_tab_manage_subjects()   
    if st.session_state.current_teacher_tab == "attendance_records":
        teacher_tab_attendance_records()           




def teacher_tab_take_attendance():
    teacher_id = st.session_state.teacher_data['teacher_id']
    st.header("Take Attendance")

    if 'attendance_images' not in st.session_state:
        st.session_state.attendance_images = []

    subjects = get_teacher_subjects(teacher_id)
    if not subjects:
        st.warning('You havent created any subjects yet ! please create one to begin !')
        return 
    subject_option = {f"{s['name']} - {s['subject_code']}": s['subject_id'] for s in subjects}

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_subject_label = st.selectbox('Select Subjects', options=list(subject_option.keys()))

    with col2:
        if st.button('Add Photos', type='primary' ,icon = ":material/photo_prints:", width='stretch'):
            add_photo_dialog()
        selected_subject_id = subject_option[selected_subject_label]

        st.divider()


    if st.session_state.attendance_images:
        st.header('Added Photos')
        gallery_cols = st.columns(4)

        for idx, img in enumerate(st.session_state.attendance_images):
            with gallery_cols[idx % 4]:
                st.image(img, width='stretch', caption=f'photo {idx+1}')

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button('Clear all Photos', width='stretch', type='tertiary', icon=':material/delete:'):
                st.session_state.attendance_images = []
                st.rerun()

        with c2:
            has_photos = bool(st.session_state.attendance_images)
            if st.button('Run Face Analysis', width='stretch', type='secondary', icon=":material/analytics:"):
                with st.spinner('Deep Scanning Clasroom Photos.......'):
                    all_detected_ids = {}

                    for idx, img in enumerate(st.session_state.attendance_images):
                        img_np = np.array(img.convert('RGB'))

                        detected, _, _ = predict_attendance(img_np)

                        if detected:
                            for sid in detected.keys():
                                student_id = int(sid)

                                all_detected_ids.setdefault(student_id, []).append(f"Photo {idx+1}")

                    enrolled_res = supabase.table('subjects_students').select("*, students(*)").eq('subject_id',selected_subject_id ).execute()
                    enrolled_students = enrolled_res.data

                    if not enrolled_students:
                        st.warning('No Student enrolled in this course') 
                    else :
                        results, attendance_to_log = [], []
                        current_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

                        for node in enrolled_students:
                            student = node['students']
                            sources = all_detected_ids.get(int(student['student_id']), [])
                            is_present= len(sources) > 0

                            results.append({
                                "Name": student['name'],
                                "ID": student['student_id'],
                                "Sources": ", ".join(sources) if is_present else "-",
                                "Status": "✅ Present" if is_present else "❌ Absent"
                            });

                            attendance_to_log.append({
                                'student_id': student['student_id'],
                                'subject_id': selected_subject_id,
                                'timestamp': current_timestamp,
                                'is_present': bool(is_present)  
                            })

                    attendance_result_dialog(pd.DataFrame(results), attendance_to_log)    
        with c3:
           if st.button('Use Voice Attendance', type='primary', width='stretch', icon=':material/mic:'):
               voice_attendance_dialog(selected_subject_id)




def teacher_tab_manage_subjects():
    teacher_id = st.session_state.teacher_data['teacher_id']
    col1, col2 = st.columns(2)
    with col1:
        st.header("Manage Subjects", width="content")
    with col2:
        if st.button("Add New Subject", width="stretch"):
            create_subject_dialog(teacher_id)

    def share_btn(sub):
        if st.button(f"Share Code: {sub['name']}", key=f"Share_{sub['subject_code']}", icon=":material/share:"):
            share_subject_dialog(sub['name'], sub['subject_code'])
        st.space()

    # List all subjects
    subjects = get_teacher_subjects(teacher_id)
    if subjects:
        for sub in subjects:
            stats = (
                ("👥", "Student", sub['total_students']),
                ("🕧", "Classes", sub['total_classes']),
            )
            subject_card(
                name=sub['name'],
                code=sub['subject_code'],
                section=sub['section'],
                stats=stats,
                footer_callback=share_btn,
                footer_callback_data=sub,
            )
    else:
        st.warning("No Subject Found")



def teacher_tab_attendance_records():
    st.header('Attendance Records')

    teacher_id = st.session_state.teacher_data['teacher_id']
    records = get_attendance_for_teacher(teacher_id)

    if not records:
        return

    data = []
    for r in records :
        ts = r.get('timestamp')

        data.append({
            "ts_group": ts.split(".")[0] if ts else None,
            "Time": datetime.fromisoformat(ts).strftime("%Y-%m-%d %I:%M %p") if ts else "N'A",
            "Subject": r['subjects']['name'],
            "Subject code":r['subjects']['subject_code'],
            "is_present": bool(r.get('is_present', False))
        })


    df = pd.DataFrame(data)    

    summary = (
        df.groupby(['ts_group', 'Time', 'Subject', 'Subject code', 'is_present'])
        .agg(
            present_count = ('is_present', 'sum'),
            total_count = ('is_present', 'count')
            ).reset_index()
    )

    summary['Attendance Stats'] = (
    "✅" + summary['present_count'].astype(str) + " /"
    + summary['total_count'].astype(str) + ' Students'
    )

    display_df = (
    summary.sort_values(by='ts_group', ascending=False)
    [['Time', 'Subject', 'Subject code', 'Attendance Stats']]
    )

    st.dataframe(display_df, width='stretch', hide_index= True)




def login_teacher(username, password):
    if not username or not password:
        return False 

    teacher = teacher_login(username, password)

    if teacher:
        st.session_state.user_roll = 'teacher'
        st.session_state.teacher_data = teacher
        st.session_state.is_logged_in = True
        return True 

    return False


def teacher_screen_login():
    c1 , c2  = st.columns(2, vertical_alignment="center", gap="xxlarge")
    with c1:
        header_dashboard()
    with c2:
        if st.button("Go Back To Home", type="secondary", key ='loginbackbtn', shortcut="control+backspace"):
            st.session_state['login_type'] = None
            st.rerun()
    
    st.header("Login Using Password",text_alignment="center")
    st.space()
    st.space()

    teacher_username = st.text_input("Enter Your Username",placeholder="ABC" )
    teacher_password = st.text_input("Enter Password", type='password', placeholder="Enter password")

    st.divider()
    btn1 , btn2 = st.columns(2)
    with btn1:
        if st.button("Login", icon=":material/passkey:", shortcut="control+enter", width="stretch"):
            if login_teacher(teacher_username, teacher_password):
                st.toast("Welcome back !", icon="👋")
                import time 
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid Username And Password")    
    with btn2:
       if st.button("Register Instead", type="primary",icon=":material/passkey:",width="stretch"):
           st.session_state.teacher_login_type = "register"



def register_teacher(teacher_username, teacher_name, teacher_password ,teacher_password_confirm):
    if not teacher_username or not teacher_name or not teacher_password:
        return False, "All Field Are Required !"
    if check_teacher_exists(teacher_username):
        return False , "Username Already Taken !"
    if teacher_password != teacher_password_confirm:
        return False, "Password Doesn,t Match"
    try:
        create_teacher(teacher_username, teacher_password, teacher_name)
        return True , "Successfully Created! Login Now"
    except Exception as e:
        return False, str(e)



def teacher_screen_register():  
    c1 , c2  = st.columns(2, vertical_alignment="center", gap="xxlarge")
    with c1:
        header_dashboard()
    with c2:
       if st.button("Go Back To Home", type="secondary", key ='loginbackbtn', shortcut="control+backspace"):
           st.session_state['login_type'] = None
           st.rerun()
    
    st.header("Register Your Teacher Profile")

    st.space()
    st.space()

    
    teacher_username = st.text_input("Enter Your Username", placeholder="ABC123", key="reg_username")
    teacher_name = st.text_input("Enter Your Name", placeholder="ABC", key="reg_name")
    teacher_password = st.text_input("Enter Password", type='password', placeholder="Enter password", key="reg_password")
    teacher_password_confirm = st.text_input("Enter Confirm Password", type="password", placeholder="Enter Confirm password", key="reg_confirm")

    st.divider()
    btn1 , btn2 = st.columns(2)
    with btn1:
        if st.button("Register", icon=":material/passkey:", shortcut="control+enter", width="stretch"): 
            success, message = register_teacher(teacher_username, teacher_name, teacher_password, teacher_password_confirm)
            if success:
                st.success(message)
                st.session_state.teacher_login_type = "login"
                import time
                time.sleep(1)
                st.rerun()
            else:
                st.error(message)

    with btn2:
       if st.button("Login Instead", type="primary",icon=":material/passkey:",width="stretch"):
           st.session_state.teacher_login_type = "login"
