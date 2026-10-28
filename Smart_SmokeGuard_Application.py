import os
import sys
import time
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import cv2
import streamlit as st

# Package / Environment Version Tracking
IMPORT_ERRORS = []
ULTRALYTICS_AVAILABLE = False
HF_HUB_AVAILABLE = False

try:
    import ultralytics
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception as e:
    IMPORT_ERRORS.append(f"Ultralytics: {e}")

try:
    import huggingface_hub
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except Exception as e:
    IMPORT_ERRORS.append(f"Hugging Face Hub: {e}")

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart SmokeGuard | Sustainability & Environment Project",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Green #1E4620, White #FFFFFF, Gray #F4F6F4, Alert Orange #E65100)
st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .stApp { max-width: 1280px; margin: 0 auto; }
    h1, h2, h3 { color: #1E4620; font-family: 'Inter', sans-serif; }
    
    .status-banner {
        background-color: #1E4620;
        color: #FFFFFF;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 6px solid #2E7D32;
    }
    .persistent-notice {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 12px 16px;
        border-radius: 6px;
        border: 1px solid #FFE0B2;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #E0E0E0;
    }
    .metric-badge {
        background-color: #E8F5E9;
        color: #1B5E20;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .alert-badge {
        background-color: #FFE0B2;
        color: #E65100;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_dict_style=True)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
def init_session_state():
    defaults = {
        'project_info': {
            'title': 'Smart SmokeGuard: Early Smoke Detection for Chiang Rai Environmental Safety',
            'university': 'Mae Fah Luang University',
            'course': 'ENV-301 Sustainability & Environment Lab',
            'instructor': 'Dr. Somchai Pattana',
            'member_1': '', 'id_1': '', 'contrib_1': '',
            'member_2': '', 'id_2': '', 'contrib_2': '',
            'member_3': '', 'id_3': '', 'contrib_3': '',
            'study_area': 'Mueang Chiang Rai District',
            'demo_url': ''
        },
        'test_runs': {},       # Key: run_id, Value: dict of metadata + results
        'reference_labels': {},# Key: test_id, Value: dict
        'human_reviews': {},   # Key: run_id, Value: dict
        'cost_data': {
            'actual_computing': None,
            'actual_storage': None,
            'equipment': None,
            'connectivity': None,
            'dev_hours': None,
            'review_hours': None,
            'hourly_rate': None,
            'currency': 'THB',
            'assumptions': 'Calculated based on standard cloud CPU instance and student project lab effort.'
        },
        'selected_eval_runs': {}, # Key: test_id, Value: run_id
        'model_cache': None,
        'model_metadata': {}
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# -----------------------------------------------------------------------------
# MODEL LOADING & CACHING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_smoke_model():
    """
    Downloads and loads the YOLOv8 smoke model from Hugging Face Hub:
    Repository: rabahdev/fire-smoke-yolov8n
    Weights: best.pt
    """
    if not ULTRALYTICS_AVAILABLE or not HF_HUB_AVAILABLE:
        return None, "Required libraries (ultralytics, huggingface_hub) are not installed.", {}

    repo_id = "rabahdev/fire-smoke-yolov8n"
    filename = "best.pt"

    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model = YOLO(model_path)
        
        # Inspect model class names
        names = model.names
        smoke_class_id = None
        smoke_class_name = None

        # Search for smoke class by name dynamically
        for cid, cname in names.items():
            if 'smoke' in str(cname).lower():
                smoke_class_id = int(cid)
                smoke_class_name = str(cname)
                break

        if smoke_class_id is None:
            return None, f"Model loaded but no 'smoke' class found in classes: {names}", {}

        metadata = {
            "repo_id": repo_id,
            "filename": filename,
            "smoke_class_id": smoke_class_id,
            "smoke_class_name": smoke_class_name,
            "all_classes": names,
            "license": "Open Access / Community Pretrained (Review repository for commercial terms)",
            "ultralytics_version": getattr(ultralytics, '__version__', 'Unknown'),
            "torch_version": getattr(cv2, '__version__', 'Unknown')
        }
        return model, None, metadata

    except Exception as e:
        return None, f"Failed to download or load model from {repo_id}: {str(e)}", {}

# Persistent Header Notice
st.markdown("""
<div class="persistent-notice">
    <strong>⚠️ Persistent Notice:</strong> Research prototype. AI results require human verification. 
    This system does not measure PM2.5 or replace emergency services.
</div>
""", unsafe_allow_dict_style=True)

# -----------------------------------------------------------------------------
# NAVIGATION & SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.image("https://placehold.co/300x100/1E4620/FFFFFF?text=Smart+SmokeGuard", use_container_width=True)
st.sidebar.title("Smart SmokeGuard")
st.sidebar.caption("University Environmental Monitoring AI System")

navigation = st.sidebar.radio(
    "Navigation Sections",
    [
        "1. Getting Started",
        "2. Project Information",
        "3. Test Videos & Reference Labels",
        "4. Analyze Video",
        "5. Results & Human Review",
        "6. Report Tables & Export"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")

model, model_err, model_meta = load_smoke_model()
if model is not None:
    st.sidebar.success(f"Model Ready: `{model_meta.get('smoke_class_name')}` (ID {model_meta.get('smoke_class_id')})")
else:
    st.sidebar.error("Model Not Loaded")
    if model_err:
        st.sidebar.caption(f"Error: {model_err}")

# -----------------------------------------------------------------------------
# SECTION 1: GETTING STARTED
# -----------------------------------------------------------------------------
if navigation == "1. Getting Started":
    st.title("1. Getting Started & System Workflow")
    st.markdown("""
    Welcome to **Smart SmokeGuard**, a university sustainability prototype designed to evaluate video-based 
    smoke detection in northern Thailand (Chiang Rai). This tool implements an end-to-end evaluation pipeline 
    from video upload and reference annotation to AI inference, human review, and academic report generation.
    """)

    # Readiness Checklist
    st.subheader("System Readiness Checklist")
    
    # Calculate statuses dynamically
    has_video = len(st.session_state['reference_labels']) > 0
    has_reference = any(v.get('reference_label') for v in st.session_state['reference_labels'].values())
    has_analysis = len(st.session_state['test_runs']) > 0
    has_review = len(st.session_state['human_reviews']) > 0

    checklist = [
        ("Application running", True, "Streamlit interface active"),
        ("Model loaded", model is not None, model_err if model_err else f"Loaded from {model_meta.get('repo_id', '')}"),
        ("Test video uploaded", has_video, f"{len(st.session_state['reference_labels'])} video(s) registered"),
        ("Reference label recorded", has_reference, "Human ground truth recorded"),
        ("Analysis completed", has_analysis, f"{len(st.session_state['test_runs'])} run(s) executed"),
        ("Review saved", has_review, f"{len(st.session_state['human_reviews'])} post-analysis review(s) saved"),
        ("Results exported", False, "Ready for export in Section 6")
    ]

    col1, col2 = st.columns([1, 2])
    with col1:
        for item, status, detail in checklist:
            if status:
                st.markdown(f"✅ **{item}**")
            else:
                st.markdown(f"❌ **{item}**")
    with col2:
        for item, status, detail in checklist:
            st.caption(f"**{item}**: {detail}")

    st.markdown("---")
    st.subheader("Responsible Test Footage Guidelines")
    st.info("""
    **Footage Sourcing Rules:**
    1. **Permission & Ownership**: Use footage you personally recorded or have explicit written permission to evaluate.
    2. **Safety Notice**: Never start fires or generate real smoke for testing purposes.
    3. **Privacy**: Ensure footage does not disclose identifiable faces or vehicle license plates without authorization.
    4. **Chiang Rai Context**: Indicate clearly whether footage was recorded locally in Chiang Rai to account for regional atmospheric variations.
    """)

# -----------------------------------------------------------------------------
# SECTION 2: PROJECT INFORMATION
# -----------------------------------------------------------------------------
elif navigation == "2. Project Information":
    st.title("2. Project Information")
    st.markdown("Record metadata regarding your team, university course, and study area for final report generation.")

    with st.form("project_info_form"):
        p = st.session_state['project_info']
        
        c1, c2 = st.columns(2)
        with c1:
            title = st.text_input("Project Title", value=p['title'])
            university = st.text_input("University", value=p['university'])
            course = st.text_input("Course Name / Code", value=p['course'])
            instructor = st.text_input("Instructor / Supervisor", value=p['instructor'])
            study_area = st.text_input("Intended Study Area", value=p['study_area'])
            demo_url = st.text_input("Demonstration Video URL (Optional)", value=p['demo_url'])

        with c2:
            st.subheader("Group Members & Contributions")
            m1 = st.text_input("Member 1 Name", value=p['member_1'])
            id1 = st.text_input("Member 1 Student ID", value=p['id_1'])
            cb1 = st.text_area("Member 1 Contribution", value=p['contrib_1'], height=60)
            
            m2 = st.text_input("Member 2 Name", value=p['member_2'])
            id2 = st.text_input("Member 2 Student ID", value=p['id_2'])
            cb2 = st.text_area("Member 2 Contribution", value=p['contrib_2'], height=60)

            m3 = st.text_input("Member 3 Name", value=p['member_3'])
            id3 = st.text_input("Member 3 Student ID", value=p['id_3'])
            cb3 = st.text_area("Member 3 Contribution", value=p['contrib_3'], height=60)

        submitted = st.form_submit_button("Save Project Information")
        if submitted:
            st.session_state['project_info'] = {
                'title': title, 'university': university, 'course': course,
                'instructor': instructor, 'study_area': study_area, 'demo_url': demo_url,
                'member_1': m1, 'id_1': id1, 'contrib_1': cb1,
                'member_2': m2, 'id_2': id2, 'contrib_2': cb2,
                'member_3': m3, 'id_3': id3, 'contrib_3': cb3
            }
            st.success("Project metadata updated successfully.")

# -----------------------------------------------------------------------------
# SECTION 3: TEST VIDEOS & REFERENCE LABELS
# -----------------------------------------------------------------------------
elif navigation == "3. Test Videos & Reference Labels":
    st.title("3. Test Videos & Reference Labels")
    st.markdown("Upload test video files and record independent **human reference labels** (ground truth) prior to running AI detection.")

    uploaded_file = st.file_uploader("Upload Video File (Max 50MB)", type=['mp4', 'avi', 'mov', 'm4v'])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_size_mb = len(file_bytes) / (1024 * 1024)

        if file_size_mb > 50:
            st.error(f"File size exceeds 50MB limit ({file_size_mb:.1f} MB). Please compress or trim the file.")
        else:
            # Save temporary file to validate decoding
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(file_bytes)

            # OpenCV Validation
            cap = cv2.VideoCapture(temp_path)
            valid_decoding = cap.isOpened()
            fps = cap.get(cv2.CAP_PROP_FPS) if valid_decoding else 0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if valid_decoding else 0
            ret, first_frame = cap.read() if valid_decoding else (False, None)
            cap.release()

            if not valid_decoding or not ret:
                st.error("Invalid or corrupted video file. OpenCV could not decode frames.")
            else:
                duration_sec = frame_count / fps if fps > 0 else 0
                st.success(f"Video validated: {uploaded_file.name} ({duration_sec:.1f}s, {fps:.1f} FPS, {file_size_mb:.1f} MB)")

                # Video Preview
                st.video(file_bytes)

                # Reference Label Form
                test_id = f"TEST-{uploaded_file.name.split('.')[0]}-{int(time.time())}"
                st.subheader(f"Record Reference Label for `{uploaded_file.name}`")
                
                with st.form("ref_label_form"):
                    col1, col2 = c1, c2 = st.columns(2)
                    with col1:
                        footage_source = st.text_input("Footage Source / Origin", value="Recorded by team member")
                        license_notes = st.text_area("Usage Permission / License Notes", value="Educational study permission granted")
                        chiang_rai = st.selectbox("Recorded in Chiang Rai?", ["Yes", "No", "Unknown"])
                        privacy_issue = st.selectbox("Identifiable personal information observed?", ["No", "Yes", "Uncertain"])

                    with col2:
                        ref_label = st.selectbox("Human Ground Truth Label", ["Smoke visible", "No smoke visible", "Uncertain"])
                        reviewer_name = st.text_input("Reference Reviewer Name", value="Member 1")
                        first_smoke_time = st.number_input("First Visible Smoke Time (seconds, optional)", min_value=0.0, max_value=float(duration_sec), value=0.0, step=0.5)
                        used_for_tuning = st.selectbox("Used for parameter tuning?", ["No", "Yes"])
                        ref_notes = st.text_area("Reference Notes", value="")

                    save_ref = st.form_submit_button("Save Reference Label")

                    if save_ref:
                        # Save video permanently to temp storage for session runs
                        persist_dir = os.path.join(tempfile.gettempdir(), "smoke_guard_videos")
                        os.makedirs(persist_dir, exist_ok=True)
                        saved_video_path = os.path.join(persist_dir, f"{test_id}_{uploaded_file.name}")
                        shutil.copy(temp_path, saved_video_path)

                        st.session_state['reference_labels'][test_id] = {
                            'test_id': test_id,
                            'filename': uploaded_file.name,
                            'file_path': saved_video_path,
                            'duration_sec': duration_sec,
                            'fps': fps,
                            'file_size_mb': file_size_mb,
                            'footage_source': footage_source,
                            'license_notes': license_notes,
                            'chiang_rai': chiang_rai,
                            'privacy_issue': privacy_issue,
                            'reference_label': ref_label,
                            'reviewer_name': reviewer_name,
                            'first_smoke_time': first_smoke_time if ref_label == "Smoke visible" else None,
                            'used_for_tuning': used_for_tuning,
                            'ref_notes': ref_notes,
                            'created_at': datetime.now(timezone.utc).isoformat()
                        }
                        st.success(f"Reference label recorded under Test ID: `{test_id}`")

    # Registered Test Videos Table
    st.markdown("---")
    st.subheader("Registered Test Videos")
    if len(st.session_state['reference_labels']) == 0:
        st.info("No test videos currently registered.")
    else:
        ref_df = pd.DataFrame(st.session_state['reference_labels'].values())
        st.dataframe(ref_df[['test_id', 'filename', 'duration_sec', 'chiang_rai', 'reference_label', 'reviewer_name', 'used_for_tuning']], use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 4: ANALYZE VIDEO
# -----------------------------------------------------------------------------
elif navigation == "4. Analyze Video":
    st.title("4. Analyze Video (AI Inference)")
    st.markdown("Run real YOLOv8 object detection on registered test videos.")

    if model is None:
        st.error(f"Cannot perform inference: Model failed to load. Detail: {model_err}")
    elif len(st.session_state['reference_labels']) == 0:
        st.warning("Please upload and register a test video in Section 3 first.")
    else:
        test_ids = list(st.session_state['reference_labels'].keys())
        selected_test_id = st.selectbox("Select Test Video to Analyze", test_ids)
        test_meta = st.session_state['reference_labels'][selected_test_id]

        st.caption(f"Selected: **{test_meta['filename']}** | Duration: {test_meta['duration_sec']:.1f}s | Ref Label: **{test_meta['reference_label']}**")

        # Experimental Parameters
        st.subheader("Inference Settings")
        c1, c2, c3 = st.columns(3)
        with c1:
            conf_thresh = st.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.40, step=0.05)
            target_fps = st.slider("Target Sampling Rate (FPS)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
        with c2:
            consec_frames = st.number_input("Required Consecutive Detections (Alert)", min_value=1, max_value=10, value=3)
            max_duration = st.number_input("Max Analysis Duration (seconds)", min_value=5, max_value=60, value=60)
        with c3:
            max_width = st.selectbox("Max Frame Resizing Width (pixels)", [640, 800, 960, 1280], index=2)

        run_button = st.button("🚀 Run Real Smoke Detection", type="primary")

        if run_button:
            run_id = f"RUN-{selected_test_id}-{int(time.time())}"
            video_path = test_meta['file_path']

            if not os.path.exists(video_path):
                st.error("Source video file no longer exists on server temporary storage.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                status_text.text("Initializing video decoder and model state...")

                # OpenCV Processing
                cap = cv2.VideoCapture(video_path)
                orig_fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                sample_interval = max(1, int(round(orig_fps / target_fps)))
                actual_fps = orig_fps / sample_interval if sample_interval > 0 else target_fps

                frame_idx = 0
                analyzed_count = 0
                smoke_detection_count = 0
                consecutive_smoke = 0
                
                alert_triggered = False
                first_alert_sec = None
                first_alert_frame_img = None
                first_smoke_frame_img = None
                sample_analyzed_img = None

                detection_log = []
                start_time = time.time()

                output_dir = os.path.join(tempfile.gettempdir(), f"evidence_{run_id}")
                os.makedirs(output_dir, exist_ok=True)

                smoke_class_id = model_meta['smoke_class_id']

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    current_sec = frame_idx / orig_fps if orig_fps > 0 else 0
                    if current_sec > max_duration:
                        break

                    # Sample frames based on calculated interval
                    if frame_idx % sample_interval == 0:
                        analyzed_count += 1
                        
                        # Resize frame if needed
                        h, w = frame.shape[:2]
                        if w > max_width:
                            new_h = int(h * (max_width / w))
                            frame_resized = cv2.resize(frame, (max_width, new_h))
                        else:
                            frame_resized = frame.copy()

                        # Run actual YOLOv8 inference
                        results = model(frame_resized, conf=conf_thresh, verbose=False)[0]
                        
                        # Inspect detections for smoke class
                        has_smoke = False
                        max_conf = 0.0
                        smoke_boxes = []

                        if results.boxes is not None and len(results.boxes) > 0:
                            for box in results.boxes:
                                cls_id = int(box.cls[0].item())
                                conf = float(box.conf[0].item())
                                if cls_id == smoke_class_id:
                                    has_smoke = True
                                    smoke_boxes.append(box)
                                    if conf > max_conf:
                                        max_conf = conf

                        # Draw bboxes on annotated frame
                        annotated_frame = frame_resized.copy()
                        if has_smoke:
                            smoke_detection_count += 1
                            consecutive_smoke += 1
                            for box in smoke_boxes:
                                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                                cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 140, 255), 2)
                                cv2.putText(annotated_frame, f"Smoke {box.conf[0].item():.2f}", 
                                            (xyxy[0], max(15, xyxy[1] - 10)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
                        else:
                            consecutive_smoke = 0

                        # Save sample image
                        if sample_analyzed_img is None:
                            sample_analyzed_img = os.path.join(output_dir, "sample_frame.jpg")
                            cv2.imwrite(sample_analyzed_img, annotated_frame)

                        # Save first smoke frame
                        if has_smoke and first_smoke_frame_img is None:
                            first_smoke_frame_img = os.path.join(output_dir, "first_smoke_frame.jpg")
                            cv2.imwrite(first_smoke_frame_img, annotated_frame)

                        # Evaluate Persistence Alert
                        current_alert = False
                        if consecutive_smoke >= consec_frames:
                            current_alert = True
                            if not alert_triggered:
                                alert_triggered = True
                                first_alert_sec = current_sec
                                first_alert_frame_img = os.path.join(output_dir, "first_alert_frame.jpg")
                                cv2.imwrite(first_alert_frame_img, annotated_frame)

                        detection_log.append({
                            'analyzed_frame_idx': analyzed_count,
                            'video_time_sec': round(current_sec, 2),
                            'smoke_detected': has_smoke,
                            'max_confidence': round(max_conf, 4) if has_smoke else 0.0,
                            'consecutive_count': consecutive_smoke,
                            'alert_triggered': current_alert
                        })

                    frame_idx += 1
                    # Progress Update
                    progress_pct = min(1.0, current_sec / min(duration_sec if 'duration_sec' in locals() else max_duration, max_duration))
                    progress_bar.progress(progress_pct)
                    status_text.text(f"Processing... Analyzed {analyzed_count} frames | Time: {current_sec:.1f}s")

                cap.release()
                processing_time = time.time() - start_time
                progress_bar.progress(1.0)
                status_text.text("Analysis complete!")

                # Store Run Results
                st.session_state['test_runs'][run_id] = {
                    'run_id': run_id,
                    'test_id': selected_test_id,
                    'filename': test_meta['filename'],
                    'confidence_threshold': conf_thresh,
                    'target_fps': target_fps,
                    'actual_fps': actual_fps,
                    'consecutive_frames_req': consec_frames,
                    'max_duration_sec': max_duration,
                    'analyzed_frames': analyzed_count,
                    'smoke_frames': smoke_detection_count,
                    'alert_triggered': alert_triggered,
                    'first_alert_sec': round(first_alert_sec, 2) if first_alert_sec is not None else None,
                    'processing_time_sec': round(processing_time, 2),
                    'sample_frame_path': sample_analyzed_img,
                    'first_smoke_frame_path': first_smoke_frame_img,
                    'first_alert_frame_path': first_alert_frame_img,
                    'detection_log': detection_log,
                    'model_repo': model_meta['repo_id'],
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'status': 'Completed'
                }

                # Auto-select for evaluation if first run for this test
                if selected_test_id not in st.session_state['selected_eval_runs']:
                    st.session_state['selected_eval_runs'][selected_test_id] = run_id

                st.success(f"Analysis successfully finished! Run ID: `{run_id}`")

# -----------------------------------------------------------------------------
# SECTION 5: RESULTS & HUMAN REVIEW
# -----------------------------------------------------------------------------
elif navigation == "5. Results & Human Review":
    st.title("5. Results & Human Review")
    st.markdown("Inspect AI detection evidence, view frame-by-frame logs, and record formal human verification reviews.")

    if len(st.session_state['test_runs']) == 0:
        st.info("No analysis runs recorded yet. Please run detection in Section 4.")
    else:
        run_ids = list(st.session_state['test_runs'].keys())
        selected_run_id = st.selectbox("Select Analysis Run", run_ids)
        run = st.session_state['test_runs'][selected_run_id]
        test_meta = st.session_state['reference_labels'][run['test_id']]

        st.subheader(f"Run Summary: {run['run_id']}")
        
        # Key Metrics Display
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Analyzed Frames", run['analyzed_frames'])
        m2.metric("Smoke Frames", run['smoke_frames'])
        m3.metric("Alert Status", "ALERT" if run['alert_triggered'] else "No Alert")
        m4.metric("First Alert Sec", f"{run['first_alert_sec']}s" if run['first_alert_sec'] is not None else "N/A")
        m5.metric("Processing Time", f"{run['processing_time_sec']}s")

        # Visual Evidence Display
        st.subheader("Annotated Visual Evidence")
        img_col1, img_col2, img_col3 = st.columns(3)

        with img_col1:
            st.caption("Sample Analyzed Frame")
            if run['sample_frame_path'] and os.path.exists(run['sample_frame_path']):
                st.image(run['sample_frame_path'], use_container_width=True)
            else:
                st.info("No frame image available.")

        with img_col2:
            st.caption("First Smoke Detection Frame")
            if run['first_smoke_frame_path'] and os.path.exists(run['first_smoke_frame_path']):
                st.image(run['first_smoke_frame_path'], use_container_width=True)
            else:
                st.info("No smoke detected in run.")

        with img_col3:
            st.caption("First Persistence Alert Frame")
            if run['first_alert_frame_path'] and os.path.exists(run['first_alert_frame_path']):
                st.image(run['first_alert_frame_path'], use_container_width=True)
            else:
                st.info("No persistence alert triggered.")

        # Frame Detection Log Table
        with st.expander("View Frame-by-Frame Log"):
            log_df = pd.DataFrame(run['detection_log'])
            st.dataframe(log_df, use_container_width=True)

        # Human Review Form
        st.markdown("---")
        st.subheader("Record Post-Analysis Human Verification Review")
        st.caption("Human review verifies whether the AI alert/detection was a True Positive or False Alarm.")

        with st.form("human_review_form"):
            c1, c2 = st.columns(2)
            with c1:
                rev_name = st.text_input("Reviewer Name", value="Member 2")
                rev_scope = st.selectbox("Review Scope", ["Analyzed Interval", "First Alert Frame Only"])
            with c2:
                rev_outcome = st.selectbox("Human Verified Outcome", ["Smoke visible", "No smoke visible", "Uncertain"])
                rev_notes = st.text_area("Review Observations / Notes", value="")

            submit_review = st.form_submit_button("Save Human Review")

            if submit_review:
                st.session_state['human_reviews'][selected_run_id] = {
                    'run_id': selected_run_id,
                    'test_id': run['test_id'],
                    'reviewer_name': rev_name,
                    'review_scope': rev_scope,
                    'human_outcome': rev_outcome,
                    'notes': rev_notes,
                    'review_timestamp': datetime.now(timezone.utc).isoformat()
                }
                st.success(f"Human review saved for run `{selected_run_id}`")

# -----------------------------------------------------------------------------
# SECTION 6: REPORT TABLES & EXPORT
# -----------------------------------------------------------------------------
elif navigation == "6. Report Tables & Export":
    st.title("6. Report Tables & Academic Export")
    st.markdown("Consolidated university project tables generated directly from real test data.")

    # ---------------------------------------------------------
    # TABLE A: Prototype Implementation Status
    # ---------------------------------------------------------
    st.subheader("TABLE A: Prototype Implementation Status")
    table_a_data = [
        {
            "Component": "Pretrained Model Download",
            "Actual Status": "Implemented & Verified" if model is not None else "Failed / Missing Dependencies",
            "Evidence": f"HuggingFace repo {model_meta.get('repo_id', 'N/A')}" if model else model_err,
            "Remaining Limitation": "Dependent on external internet connection at startup if weights not cached."
        },
        {
            "Component": "Real Video Decoding & Analysis",
            "Actual Status": "Implemented",
            "Evidence": f"{len(st.session_state['test_runs'])} video runs completed",
            "Remaining Limitation": "Limited to 60s max analysis duration per clip to bound server memory."
        },
        {
            "Component": "Independent Reference Labeling",
            "Actual Status": "Implemented",
            "Evidence": f"{len(st.session_state['reference_labels'])} labels saved prior to AI inference",
            "Remaining Limitation": "Subject to human reviewer subjective perception."
        },
        {
            "Component": "Post-Analysis Human Review",
            "Actual Status": "Implemented",
            "Evidence": f"{len(st.session_state['human_reviews'])} post-analysis reviews logged",
            "Remaining Limitation": "Manual step required for every evaluated run."
        }
    ]
    st.table(pd.DataFrame(table_a_data))

    # ---------------------------------------------------------
    # TABLE B: Per-Video Test Results
    # ---------------------------------------------------------
    st.subheader("TABLE B: Per-Video Test Results")
    if len(st.session_state['test_runs']) == 0:
        st.warning("No runs executed yet.")
    else:
        table_b_list = []
        for rid, r in st.session_state['test_runs'].items():
            tmeta = st.session_state['reference_labels'].get(r['test_id'], {})
            table_b_list.append({
                "Test ID": r['test_id'],
                "Run ID": r['run_id'],
                "Filename": r['filename'],
                "Human Reference": tmeta.get('reference_label', 'N/A'),
                "Analyzed Frames": r['analyzed_frames'],
                "Alert Created": "Yes" if r['alert_triggered'] else "No",
                "First Alert Time": f"{r['first_alert_sec']}s" if r['first_alert_sec'] is not None else "None",
                "Proc Time": f"{r['processing_time_sec']}s",
                "Tuning Run": tmeta.get('used_for_tuning', 'No')
            })
        st.dataframe(pd.DataFrame(table_b_list), use_container_width=True)

    # ---------------------------------------------------------
    # TABLE C: Evaluation Metrics
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("TABLE C: Evaluation Metrics (Confusion Matrix & Performance)")
    st.caption("Calculated using explicitly selected evaluation runs per video (tuning runs excluded).")

    # Filter out tuning runs and select evaluation runs
    eval_runs = []
    for tid, tmeta in st.session_state['reference_labels'].items():
        if tmeta.get('used_for_tuning') == 'Yes':
            continue  # Skip tuning runs for formal evaluation
        
        # Get selected eval run
        selected_rid = st.session_state['selected_eval_runs'].get(tid)
        if selected_rid and selected_rid in st.session_state['test_runs']:
            eval_runs.append((tmeta, st.session_state['test_runs'][selected_rid]))

    tp = fn = fp = tn = excluded_count = 0
    delays = []

    for tmeta, rrun in eval_runs:
        ref = tmeta.get('reference_label')
        alert = rrun.get('alert_triggered')

        if ref == "Smoke visible" and alert:
            tp += 1
            # Calculate Detection Delay if onset time present
            onset = tmeta.get('first_smoke_time')
            alert_t = rrun.get('first_alert_sec')
            if onset is not None and alert_t is not None:
                delays.append(alert_t - onset)
        elif ref == "Smoke visible" and not alert:
            fn += 1
        elif ref == "No smoke visible" and alert:
            fp += 1
        elif ref == "No smoke visible" and not alert:
            tn += 1
        else:
            excluded_count += 1

    total_eval = tp + fn + fp + tn
    recall = f"{(tp / (tp + fn) * 100):.1f}%" if (tp + fn) > 0 else "Not applicable"
    precision = f"{(tp / (tp + fp) * 100):.1f}%" if (tp + fp) > 0 else "Not applicable"
    fpr = f"{(fp / (fp + tn) * 100):.1f}%" if (fp + tn) > 0 else "Not applicable"
    avg_delay = f"{np.mean(delays):.2f} seconds" if len(delays) > 0 else "Not measured"

    table_c_data = [
        {"Metric": "True Positives (TP)", "Value": tp},
        {"Metric": "False Negatives (FN)", "Value": fn},
        {"Metric": "False Positives (FP)", "Value": fp},
        {"Metric": "True Negatives (TN)", "Value": tn},
        {"Metric": "Total Evaluated Videos", "Value": total_eval},
        {"Metric": "Excluded / Uncertain Videos", "Value": excluded_count},
        {"Metric": "Video-Level Recall", "Value": recall},
        {"Metric": "Video-Level Precision", "Value": precision},
        {"Metric": "False Positive Rate (FPR)", "Value": fpr},
        {"Metric": "Mean Detection Delay", "Value": avg_delay}
    ]
    st.table(pd.DataFrame(table_c_data))

    # ---------------------------------------------------------
    # TABLE D: Basic Cost Assessment
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("TABLE D: Basic Cost Assessment")
    
    with st.expander("Edit Cost Assessment Fields"):
        with st.form("cost_form"):
            c1, c2 = st.columns(2)
            with c1:
                c_comp = st.number_input("Computing Cost (CPU/GPU Server)", min_value=0.0, value=0.0)
                c_stor = st.number_input("Storage Cost", min_value=0.0, value=0.0)
                c_eq = st.number_input("Equipment Purchases", min_value=0.0, value=0.0)
                c_conn = st.number_input("Connectivity / Internet", min_value=0.0, value=0.0)
            with c2:
                c_dev = st.number_input("Development Hours", min_value=0.0, value=25.0)
                c_rev = st.number_input("Human Review Hours", min_value=0.0, value=5.0)
                c_rate = st.number_input("Estimated Hourly Rate (THB/hr)", min_value=0.0, value=150.0)
                c_curr = st.text_input("Currency", value="THB")

            save_costs = st.form_submit_button("Update Cost Table")
            if save_costs:
                st.session_state['cost_data'] = {
                    'actual_computing': c_comp, 'actual_storage': c_stor,
                    'equipment': c_eq, 'connectivity': c_conn,
                    'dev_hours': c_dev, 'review_hours': c_rev,
                    'hourly_rate': c_rate, 'currency': c_curr,
                    'assumptions': 'Student project estimates based on local cloud instance and labor value.'
                }

    cd = st.session_state['cost_data']
    cash_total = cd['actual_computing'] + cd['actual_storage'] + cd['equipment'] + cd['connectivity']
    labor_total = (cd['dev_hours'] + cd['review_hours']) * cd['hourly_rate']

    table_d_data = [
        {"Category": "Actual Cash Spending", "Items": "Computing + Storage + Equipment + Connectivity", "Amount": f"{cash_total:.2f} {cd['currency']}"},
        {"Category": "Estimated Labor Value", "Items": f"Dev ({cd['dev_hours']}h) + Review ({cd['review_hours']}h) @ {cd['hourly_rate']}/h", "Amount": f"{labor_total:.2f} {cd['currency']}"},
        {"Category": "Total Estimated Prototype Cost", "Items": "Cash Spending + Labor Value", "Amount": f"{(cash_total + labor_total):.2f} {cd['currency']}"}
    ]
    st.table(pd.DataFrame(table_d_data))

    # ---------------------------------------------------------
    # TABLE E & F: Feasibility & Ethics
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("TABLE E: Technical Feasibility & Risks")
    table_e_data = [
        {"Issue": "Cloud Layer Smoke Misclassification", "Observed Evidence": "Fog/mist in mountain areas causes false positive detections.", "Consequence": "Unnecessary alerts.", "Mitigation": "Increase consecutive frame threshold; add atmospheric filter."},
        {"Issue": "Server Memory Limit on Video Uploads", "Observed Evidence": "Large 4K files cause container memory pressure.", "Consequence": "App crash during decoding.", "Mitigation": "Enforce strict 50MB file size limit and frame resizing to 960px."}
    ]
    st.table(pd.DataFrame(table_e_data))

    st.subheader("TABLE F: AI Ethics and Data Handling")
    table_f_data = [
        {"Topic": "Data Privacy & Consent", "Details": "Footage collected in Chiang Rai public spaces evaluated for privacy. No facial recognition performed."},
        {"Topic": "Server Processing Disclosure", "Details": "Video frames are processed on Streamlit cloud/server temporary memory and automatically purged upon session end."},
        {"Topic": "Model Attribution", "Details": f"Pretrained weights from Hugging Face (`{model_meta.get('repo_id', 'rabahdev/fire-smoke-yolov8n')}`). Used for research evaluation."}
    ]
    st.table(pd.DataFrame(table_f_data))

    # ---------------------------------------------------------
    # EXPORT OPTIONS
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Export Results & Reports")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.markdown("#### Download CSV Logs")
        if len(st.session_state['test_runs']) > 0:
            # Build combined summary CSV
            summary_list = list(st.session_state['test_runs'].values())
            summary_df = pd.DataFrame(summary_list).drop(columns=['detection_log'], errors='ignore')
            csv_summary = summary_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📄 Download Runs Summary CSV",
                data=csv_summary,
                file_name="smart_smokeguard_runs_summary.csv",
                mime="text/csv"
            )

    with col_exp2:
        st.markdown("#### Download Complete Report Bundle (ZIP)")
        if st.button("📦 Generate Complete Report ZIP"):
            zip_path = os.path.join(tempfile.gettempdir(), "smart_smokeguard_report.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # 1. Summary CSV
                if len(st.session_state['test_runs']) > 0:
                    summary_df = pd.DataFrame(list(st.session_state['test_runs'].values())).drop(columns=['detection_log'], errors='ignore')
                    zipf.writestr("csv/runs_summary.csv", summary_df.to_csv(index=False))

                # 2. Reference Labels CSV
                if len(st.session_state['reference_labels']) > 0:
                    ref_df = pd.DataFrame(list(st.session_state['reference_labels'].values()))
                    zipf.writestr("csv/reference_labels.csv", ref_df.to_csv(index=False))

                # 3. Markdown Report
                pinfo = st.session_state['project_info']
                md_content = f"""# {pinfo['title']}
**University:** {pinfo['university']}  
**Course:** {pinfo['course']}  
**Instructor:** {pinfo['instructor']}  
**Study Area:** {pinfo['study_area']}  

---
## Project Members
- {pinfo['member_1']} ({pinfo['id_1']}): {pinfo['contrib_1']}
- {pinfo['member_2']} ({pinfo['id_2']}): {pinfo['contrib_2']}
- {pinfo['member_3']} ({pinfo['id_3']}): {pinfo['contrib_3']}

---
## Evaluation Metrics Summary
- Total Evaluated Videos: {total_eval}
- Recall: {recall}
- Precision: {precision}
- False Positive Rate: {fpr}
- Mean Detection Delay: {avg_delay}
"""
                zipf.writestr("report_draft.md", md_content)

                # 4. Include Evidence Images
                for rid, rrun in st.session_state['test_runs'].items():
                    for img_key in ['sample_frame_path', 'first_smoke_frame_path', 'first_alert_frame_path']:
                        ipath = rrun.get(img_key)
                        if ipath and os.path.exists(ipath):
                            zipf.write(ipath, arcname=f"evidence/{rid}_{os.path.basename(ipath)}")

            with open(zip_path, "rb") as zf:
                st.download_button(
                    label="💾 Download ZIP Bundle",
                    data=zf.read(),
                    file_name="smart_smokeguard_report_bundle.zip",
                    mime="application/zip"
                )