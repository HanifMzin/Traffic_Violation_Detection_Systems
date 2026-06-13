import streamlit as st
import pandas as pd
import os
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Traffic Violation Detection Dashboard",
    page_icon="🚦",
    layout="wide"
)

# =========================================================
# PATH CONFIGURATION
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

CSV_CANDIDATES = [
    os.path.join(CURRENT_DIR, "records", "violation_log.csv"),
    os.path.join(SCRIPT_DIR, "records", "violation_log.csv"),
    os.path.join(PARENT_DIR, "records", "violation_log.csv"),
    os.path.join(CURRENT_DIR, "logs", "violations_log.csv"),
    os.path.join(SCRIPT_DIR, "logs", "violations_log.csv"),
    os.path.join(PARENT_DIR, "logs", "violations_log.csv"),
]

LOGO_CANDIDATES = [
    os.path.join(CURRENT_DIR, "Unikl3.png"),
    os.path.join(SCRIPT_DIR, "Unikl3.png"),
    os.path.join(PARENT_DIR, "Unikl3.png"),

    os.path.join(CURRENT_DIR, "assets", "Unikl3.png"),
    os.path.join(SCRIPT_DIR, "assets", "Unikl3.png"),
    os.path.join(PARENT_DIR, "assets", "Unikl3.png"),
]

LOGO_CANDIDATES2 = [
    os.path.join(CURRENT_DIR, "Unikl2.jpg"),
    os.path.join(SCRIPT_DIR, "Unikl2.jpg"),
    os.path.join(PARENT_DIR, "Unikl2.jpg"),

    os.path.join(CURRENT_DIR, "assets", "Unikl2.jpg"),
    os.path.join(SCRIPT_DIR, "assets", "Unikl2.jpg"),
    os.path.join(PARENT_DIR, "assets", "Unikl2.jpg"),
]


def find_existing_file(path_list):
    for path in path_list:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


CSV_LOG_PATH = find_existing_file(CSV_CANDIDATES)
LOGO_PATH = find_existing_file(LOGO_CANDIDATES)
LOGO_PATH2 = find_existing_file(LOGO_CANDIDATES2)

if CSV_LOG_PATH is not None:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(CSV_LOG_PATH), ".."))
else:
    PROJECT_ROOT = CURRENT_DIR


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #b8b8b8;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #ffffff;
    }

    .metric-card {
        background-color: #1e1e1e;
        padding: 18px;
        border-radius: 15px;
        border: 1px solid #333333;
        text-align: center;
        min-height: 110px;
    }

    .metric-label {
        color: #b8b8b8;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
    }

    .evidence-title {
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .evidence-info {
        font-size: 13px;
        color: #b8b8b8;
        margin-bottom: 8px;
    }

    .warning-box {
        background-color: #3f3f0f;
        color: #fef08a;
        padding: 14px;
        border-radius: 10px;
        font-size: 15px;
    }

    .success-box {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 14px;
        border-radius: 10px;
        font-size: 15px;
    }

    .status-detected {
        background-color: #00c853;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .status-unknown {
        background-color: #ff3131;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .status-general {
        background-color: #555555;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_value(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.lower() in ["nan", "none"]:
        return ""

    return value


def normalize_path(path_value):
    if pd.isna(path_value):
        return ""

    path_value = str(path_value).strip()

    if path_value.lower() in ["", "nan", "none", "unknown"]:
        return ""

    path_value = path_value.replace("\\", "/")

    return path_value


def resolve_file_path(path_value):
    """
    Function ini fix masalah image not found.
    Dia cari path dari:
    1. Absolute path
    2. Project root
    3. Current directory
    4. Script directory
    5. Parent directory
    """

    path_value = normalize_path(path_value)

    if path_value == "":
        return None

    if os.path.isabs(path_value):
        if os.path.exists(path_value):
            return path_value
        return None

    base_dirs = [
        PROJECT_ROOT,
        CURRENT_DIR,
        SCRIPT_DIR,
        PARENT_DIR,
    ]

    for base in base_dirs:
        possible_path = os.path.abspath(os.path.join(base, path_value))

        if os.path.exists(possible_path):
            return possible_path

    return None


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def format_float(value):
    return f"{safe_float(value):.2f}"


def format_track_id(value):
    if pd.isna(value):
        return ""

    try:
        value_float = float(value)

        if value_float.is_integer():
            return str(int(value_float))

        return str(value_float)

    except Exception:
        return str(value)


def is_plate_detected(plate_number):
    plate_number = clean_value(plate_number)

    if plate_number == "":
        return False

    if plate_number.lower() in ["unknown", "not detected", "none", "nan"]:
        return False

    return True


def get_plate_status(plate_number):
    if is_plate_detected(plate_number):
        return "Detected"
    return "Unknown"


def display_metric_card(title, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_image(path_value, caption, not_found_text):
    resolved_path = resolve_file_path(path_value)

    if resolved_path is not None:
        try:
            image = Image.open(resolved_path)
            st.image(image, caption=caption, use_container_width=True)
        except Exception as e:
            st.error(f"Unable to open image: {e}")
    else:
        st.markdown(
            f"""
            <div class="warning-box">
                {not_found_text}
            </div>
            """,
            unsafe_allow_html=True
        )


def make_record_label(index, row):
    date = clean_value(row.get("Date", ""))
    time_value = clean_value(row.get("Time", ""))
    violation_type = clean_value(row.get("Violation Type", ""))
    vehicle_type = clean_value(row.get("Vehicle Type", ""))
    plate_number = clean_value(row.get("Plate Number", ""))

    if plate_number == "":
        plate_number = "Unknown"

    return f"{index} | {date} {time_value} | {violation_type} | {vehicle_type} | Plate: {plate_number}"


def show_status_badge(text, status_type="general"):
    if status_type == "detected":
        class_name = "status-detected"
    elif status_type == "unknown":
        class_name = "status-unknown"
    else:
        class_name = "status-general"

    st.markdown(
        f"""
        <span class="{class_name}">{text}</span>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data(csv_path):
    if csv_path is None:
        return pd.DataFrame()

    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    return df


def standardize_columns(df):
    """
    Support both old CSV and new CSV.
    Old dashboard used:
    - Violation
    - Confidence (%)
    - Traffic Light Status

    New system uses:
    - Violation Type
    - Vehicle Type
    - Plate Number
    - Plate Detection Confidence
    - OCR Confidence
    - Evidence Image Path
    - Vehicle Crop Path
    - Plate Crop Path
    """

    if "Violation Type" not in df.columns and "Violation" in df.columns:
        df["Violation Type"] = df["Violation"]

    if "Plate Detection Confidence" not in df.columns:
        if "Confidence (%)" in df.columns:
            df["Plate Detection Confidence"] = df["Confidence (%)"]
        else:
            df["Plate Detection Confidence"] = 0.0

    if "OCR Confidence" not in df.columns:
        df["OCR Confidence"] = 0.0

    required_columns = [
        "Date",
        "Time",
        "Violation Type",
        "Vehicle Type",
        "Track ID",
        "Plate Number",
        "Plate Detection Confidence",
        "OCR Confidence",
        "Evidence Image Path",
        "Vehicle Crop Path",
        "Plate Crop Path",
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    df["Plate Detection Confidence"] = pd.to_numeric(
        df["Plate Detection Confidence"],
        errors="coerce"
    ).fillna(0.0)

    df["OCR Confidence"] = pd.to_numeric(
        df["OCR Confidence"],
        errors="coerce"
    ).fillna(0.0)

    df["Plate Status"] = df["Plate Number"].apply(get_plate_status)

    if "Date" in df.columns:
        df["Date Parsed"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


df = load_data(CSV_LOG_PATH)

if not df.empty:
    df = standardize_columns(df)


# =========================================================
# SIDEBAR
# =========================================================

if LOGO_PATH is not None:
    st.sidebar.image(LOGO_PATH, width=280)
else:
    st.sidebar.warning("Logo 1 not found.")

st.sidebar.title("🚦 Dashboard Filter")
st.sidebar.caption("By Hanif Mohmmad Zin")

st.sidebar.divider()

if st.sidebar.button("🔄 Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

with st.sidebar.expander("System Path Info"):
    st.write("CSV path:")
    st.code(str(CSV_LOG_PATH), language="text")

    st.write("Project root:")
    st.code(str(PROJECT_ROOT), language="text")


# =========================================================
# HEADER
# =========================================================

header_col1, header_col2 = st.columns([0.12, 0.88])

with header_col1:
    if LOGO_PATH2 is not None:
        st.image(LOGO_PATH2, width=300)
    else:
        st.markdown("## 🚦")

with header_col2:
    st.markdown(
        """
        <div class="main-title">DEVELOPMENT OF TRAFFIC VIOLATION DETECTION 
        FOR ALL TYPES OF MOTOR VEHICLE</div>
        <div class="subtitle">
        >> Helmet Violation Detection  >> Red Light Violation Detection >> License Plate Recognition </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# CHECK DATA
# =========================================================

if df.empty:
    st.warning("No violation log found. Please run `main.py` first to generate CSV log and evidence images.")

    st.info("Dashboard tried to find CSV at:")

    for path in CSV_CANDIDATES:
        st.code(path, language="text")

    st.stop()


# =========================================================
# FILTERS
# =========================================================

filtered_df = df.copy()

# Date filter
date_options = ["All"] + sorted(df["Date"].dropna().astype(str).unique().tolist())
selected_date = st.sidebar.selectbox("Date", date_options)

if selected_date != "All":
    filtered_df = filtered_df[filtered_df["Date"].astype(str) == selected_date]

# Violation type filter
violation_options = ["All"] + sorted(df["Violation Type"].dropna().astype(str).unique().tolist())
selected_violation = st.sidebar.selectbox("Violation Type", violation_options)

if selected_violation != "All":
    filtered_df = filtered_df[filtered_df["Violation Type"].astype(str) == selected_violation]

# Vehicle type filter
vehicle_options = ["All"] + sorted(df["Vehicle Type"].dropna().astype(str).unique().tolist())
selected_vehicle = st.sidebar.selectbox("Vehicle Type", vehicle_options)

if selected_vehicle != "All":
    filtered_df = filtered_df[filtered_df["Vehicle Type"].astype(str) == selected_vehicle]

# Plate status filter
plate_status_options = ["All", "Detected", "Unknown"]
selected_plate_status = st.sidebar.selectbox("Plate Status", plate_status_options)

if selected_plate_status != "All":
    filtered_df = filtered_df[filtered_df["Plate Status"] == selected_plate_status]

# OCR confidence filter
ocr_range = st.sidebar.slider(
    "OCR Confidence Range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.01
)

filtered_df = filtered_df[
    (filtered_df["OCR Confidence"] >= ocr_range[0]) &
    (filtered_df["OCR Confidence"] <= ocr_range[1])
]


# =========================================================
# METRICS
# =========================================================

total_violations = len(filtered_df)

helmet_violations = len(
    filtered_df[
        filtered_df["Violation Type"].astype(str).str.lower().str.contains("helmet", na=False)
    ]
)

red_light_violations = len(
    filtered_df[
        filtered_df["Violation Type"].astype(str).str.lower().str.contains("red light", na=False)
    ]
)

plate_detected_count = len(filtered_df[filtered_df["Plate Status"] == "Detected"])
unknown_plate_count = len(filtered_df[filtered_df["Plate Status"] == "Unknown"])

average_plate_conf = filtered_df["Plate Detection Confidence"].mean() if total_violations > 0 else 0
average_ocr_conf = filtered_df["OCR Confidence"].mean() if total_violations > 0 else 0

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    display_metric_card("Total Violations", total_violations)

with m2:
    display_metric_card("Helmet Violations", helmet_violations)

with m3:
    display_metric_card("Red Light Violations", red_light_violations)

with m4:
    display_metric_card("Plate Detected", plate_detected_count)

with m5:
    display_metric_card("Unknown Plate", unknown_plate_count)

m6, m7 = st.columns(2)

with m6:
    display_metric_card("Avg Plate Detection Conf.", f"{average_plate_conf:.2f}")

with m7:
    display_metric_card("Avg OCR Conf.", f"{average_ocr_conf:.2f}")


st.divider()


# =========================================================
# ANALYTICS
# =========================================================

st.markdown('<div class="section-title">📊 Violation Analytics</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Violation Type Summary")

    if not filtered_df.empty:
        violation_count = filtered_df["Violation Type"].value_counts()
        st.bar_chart(violation_count)
    else:
        st.info("No data available.")

with chart_col2:
    st.subheader("Vehicle Type Summary")

    if not filtered_df.empty:
        vehicle_count = filtered_df["Vehicle Type"].value_counts()
        st.bar_chart(vehicle_count)
    else:
        st.info("No data available.")

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("Plate Status Summary")

    if not filtered_df.empty:
        plate_count = filtered_df["Plate Status"].value_counts()
        st.bar_chart(plate_count)
    else:
        st.info("No data available.")

with chart_col4:
    st.subheader("OCR Confidence Trend")

    if not filtered_df.empty:
        st.line_chart(filtered_df["OCR Confidence"].reset_index(drop=True))
    else:
        st.info("No OCR data available.")


st.divider()


# =========================================================
# PAGINATED EVIDENCE RECORDS
# =========================================================

st.markdown('<div class="section-title">📸 Evidence Records</div>', unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# Reset index supaya numbering kemas
viewer_df = filtered_df.iloc[::-1].reset_index(drop=True)

RECORDS_PER_PAGE = 5
total_records = len(viewer_df)
total_pages = (total_records + RECORDS_PER_PAGE - 1) // RECORDS_PER_PAGE

# Session state untuk page number
if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# Kalau filter berubah dan page lebih besar daripada total page, reset balik
if st.session_state.current_page > total_pages:
    st.session_state.current_page = 1

# Calculate page slice
start_idx = (st.session_state.current_page - 1) * RECORDS_PER_PAGE
end_idx = start_idx + RECORDS_PER_PAGE

page_df = viewer_df.iloc[start_idx:end_idx]

st.divider()

# Display records
for idx, row in page_df.iterrows():
    record_number = start_idx + idx + 1

    with st.container(border=True):
        # Header record
        header_col1, header_col2, header_col3 = st.columns([2, 2, 1])

        with header_col1:
            st.markdown(
                f"""
                <div class="evidence-title">
                    Record #{record_number} - {clean_value(row.get("Violation Type", "Violation"))}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="evidence-info">
                    Date: {clean_value(row.get("Date", "-"))}<br>
                    Time: {clean_value(row.get("Time", "-"))}<br>
                    Vehicle Type: {clean_value(row.get("Vehicle Type", "-"))}
                </div>
                """,
                unsafe_allow_html=True
            )

        with header_col2:
            st.markdown(
                f"""
                <div class="evidence-info">
                    Track ID: {format_track_id(row.get("Track ID", ""))}<br>
                    Plate Number: {clean_value(row.get("Plate Number", "Unknown"))}<br>
                    Plate Detection Confidence: {format_float(row.get("Plate Detection Confidence", 0.0))}<br>
                    OCR Confidence: {format_float(row.get("OCR Confidence", 0.0))}
                </div>
                """,
                unsafe_allow_html=True
            )

        with header_col3:
            st.markdown("**Plate Status:**")

            if row.get("Plate Status", "Unknown") == "Detected":
                show_status_badge("Detected", "detected")
            else:
                show_status_badge("Unknown", "unknown")

        st.markdown("---")

        # Image display
        img_col1, img_col2, img_col3 = st.columns(3)

        with img_col1:
            st.subheader("Evidence Image")
            display_image(
                row.get("Evidence Image Path", ""),
                "Evidence Image",
                "Evidence image not found."
            )

        with img_col2:
            st.subheader("Vehicle Crop")
            display_image(
                row.get("Vehicle Crop Path", ""),
                "Vehicle Crop",
                "Vehicle crop not found."
            )

        with img_col3:
            st.subheader("Plate Crop")

            if row.get("Plate Status", "Unknown") == "Unknown":
                st.markdown(
                    """
                    <div class="warning-box">
                        Plate crop not available because plate was not detected or OCR returned Unknown.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                display_image(
                    row.get("Plate Crop Path", ""),
                    "Plate Crop",
                    "Plate crop not found."
                )


# =========================================================
# PAGINATION CONTROLS - BELOW EVIDENCE RECORDS
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

page_col1, page_col2, page_col3 = st.columns([1, 2, 1])

with page_col1:
    if st.button("⬅️ Previous", disabled=st.session_state.current_page <= 1):
        st.session_state.current_page -= 1
        st.rerun()

with page_col2:
    st.markdown(
        f"""
        <div style="text-align:center; font-size:18px; font-weight:700;">
            Page {st.session_state.current_page} of {total_pages}
            <br>
            <span style="font-size:14px; color:#b8b8b8;">
                Showing {RECORDS_PER_PAGE} records per page
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

with page_col3:
    if st.button("Next ➡️", disabled=st.session_state.current_page >= total_pages):
        st.session_state.current_page += 1
        st.rerun()


st.divider()


# =========================================================
# TABLE SECTION
# =========================================================

st.markdown('<div class="section-title">📄 Violation Log Table</div>', unsafe_allow_html=True)

display_df = filtered_df.copy()

if "Date Parsed" in display_df.columns:
    display_df = display_df.drop(columns=["Date Parsed"])

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# DOWNLOAD BUTTON
# =========================================================

csv_data = display_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered CSV",
    data=csv_data,
    file_name="filtered_violation_log.csv",
    mime="text/csv"
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="success-box">
        Dashboard loaded successfully.
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Final Year Project Dashboard | Traffic Violation Detection System using YOLO11"
)


# To Run Dashboard Using Terminal:
# python -m streamlit run Dashboard.py