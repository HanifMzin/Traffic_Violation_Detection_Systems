from ultralytics import YOLO
import cv2
import time
import os
import math
from datetime import datetime
import easyocr
import re
import csv

# ==============================
# CONFIGURATION
# ==============================

HELMET_MODEL_PATH = "Models/Helmet.pt"
TRAFFIC_LIGHT_MODEL_PATH = "Models/Traffic_Light.pt"
VEHICLE_MODEL_PATH = "yolo11n.pt"
PLATE_MODEL_PATH = "Models/Plate.pt"

VIDEO_PATH = "videos/Testing 5.mp4"
USE_WEBCAM = True

CONF_THRESHOLD_HELMET = 0.35
CONF_THRESHOLD_TRAFFIC_LIGHT = 0.80
CONF_THRESHOLD_VEHICLE = 0.35
CONF_THRESHOLD_PLATE = 0.10

FRAME_WIDTH = 1080
FRAME_HEIGHT = 720
USE_ORIGINAL_VIDEO_SIZE = False

# ==============================
# OCR SETTINGS
# ==============================

OCR_LANGUAGES = ["en"]
OCR_CONF_THRESHOLD = 0.10

# ==============================
# DISPLAY SETTINGS
# ==============================

SHOW_NORMAL_DETECTIONS = True
SHOW_STOP_LINE = True

# ==============================
# EVIDENCE CAPTURE SETTINGS
# ==============================

EVIDENCE_DIR = "evidence"
HELMET_EVIDENCE_DIR = os.path.join(EVIDENCE_DIR, "helmet_violation")
RED_LIGHT_EVIDENCE_DIR = os.path.join(EVIDENCE_DIR, "red_light_violation")
MULTIPLE_EVIDENCE_DIR = os.path.join(EVIDENCE_DIR, "multiple_violation")

PLATE_DIR = "plates"
VEHICLE_CROP_DIR = os.path.join(PLATE_DIR, "vehicle_crops")
CROPPED_PLATE_DIR = os.path.join(PLATE_DIR, "cropped_plates")

HELMET_SAVE_COOLDOWN = 1.0
RED_LIGHT_SAVE_COOLDOWN = 2.0

helmet_evidence_saved = False
last_helmet_save_time = 0
last_red_light_save_time = 0

# ==============================
# CSV LOGGING SETTINGS
# ==============================

RECORDS_DIR = "records"
CSV_LOG_PATH = os.path.join(RECORDS_DIR, "violation_log.csv")

CSV_COLUMNS = [
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
    "Plate Crop Path"
]

# ==============================
# HELMET VIOLATION SETTINGS
# ==============================

HELMET_CONFIRM_FRAMES = 5
HELMET_CLEAR_FRAMES = 3

helmet_violation_counter = 0
helmet_clear_counter = 0
helmet_violation_active = False
last_confirmed_helmet_violations = []

PERSON_MARGIN = 20
MOTORCYCLE_X_MARGIN = 80
MOTORCYCLE_TOP_MARGIN = 160
MOTORCYCLE_BOTTOM_MARGIN = 80

# ==============================
# RED LIGHT VIOLATION SETTINGS
# ==============================

#The more < number the line going up
#The more > number the line going down
STOP_LINE_Y = 500

# "down" = vehicle bergerak dari atas frame ke bawah frame
# "up"   = vehicle bergerak dari bawah frame ke atas frame
ROAD_DIRECTION = "up"

RED_LIGHT_DISPLAY_FRAMES = 30
TRAFFIC_LIGHT_MEMORY_FRAMES = 10

last_known_light = "unknown"
last_known_light_conf = 0.0
traffic_light_missing_counter = 0

TRACK_MAX_DISTANCE = 120
TRACK_MAX_MISSING = 10

vehicle_tracks = {}
next_track_id = 1
active_red_light_events = []

# ==============================
# DUPLICATE FILTER SETTINGS
# ==============================

HELMET_DUPLICATE_IOU = 0.35
TRAFFIC_DUPLICATE_IOU = 0.35
VEHICLE_DUPLICATE_IOU = 0.25
HELMET_CONFLICT_IOU = 0.30

# ==============================
# CHECK MODEL FILES
# ==============================

if not os.path.exists(HELMET_MODEL_PATH):
    print(f"ERROR: Helmet model not found at {HELMET_MODEL_PATH}")
    print("Please make sure Helmet.pt is inside the Models folder.")
    exit()

if not os.path.exists(TRAFFIC_LIGHT_MODEL_PATH):
    print(f"ERROR: Traffic light model not found at {TRAFFIC_LIGHT_MODEL_PATH}")
    print("Please make sure Traffic_Light.pt is inside the Models folder.")
    exit()

if not os.path.exists(PLATE_MODEL_PATH):
    print(f"ERROR: Plate model not found at {PLATE_MODEL_PATH}")
    print("Please make sure Plate.pt is inside the Models folder.")
    exit()

# ==============================
# LOAD MODELS
# ==============================

print("Loading helmet detection model...")
model_helmet = YOLO(HELMET_MODEL_PATH)
print("Helmet model loaded successfully.")
print("Helmet model classes:")
print(model_helmet.names)

print("\nLoading traffic light detection model...")
model_traffic = YOLO(TRAFFIC_LIGHT_MODEL_PATH)
print("Traffic light model loaded successfully.")
print("Traffic light model classes:")
print(model_traffic.names)

print("\nLoading YOLO vehicle/person model...")
model_vehicle = YOLO(VEHICLE_MODEL_PATH)
print("Vehicle/person model loaded successfully.")
print("Vehicle model classes:")
print(model_vehicle.names)

print("\nLoading license plate model...")
model_plate = YOLO(PLATE_MODEL_PATH)
print("License plate model loaded successfully.")
print("Plate model classes:")
print(model_plate.names)

print("\nLoading OCR reader...")
ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=True)
print("OCR reader loaded successfully.")

# ==============================
# VIDEO / WEBCAM SOURCE
# ==============================

if USE_WEBCAM:
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
else:
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found at {VIDEO_PATH}")
        print("Please place your video inside the videos folder.")
        exit()

    cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video/webcam.")
    exit()

# ==============================
# CLASS FILTERS
# ==============================

vehicle_related_classes = [
    "person",
    "motorcycle",
    "motorbike",
    "car",
    "truck",
    "bus"
]

motorcycle_class_names = ["motorcycle", "motorbike"]

helmet_class_names = [
    "with helmet",
    "without helmet"
]

traffic_light_class_names = [
    "green light",
    "red light",
    "yellow light"
]

red_light_vehicle_classes = [
    "motorcycle",
    "motorbike",
    "car",
    "truck",
    "bus"
]

# ==============================
# FILE / CSV FUNCTIONS
# ==============================

def create_output_folders():
    os.makedirs(HELMET_EVIDENCE_DIR, exist_ok=True)
    os.makedirs(RED_LIGHT_EVIDENCE_DIR, exist_ok=True)
    os.makedirs(MULTIPLE_EVIDENCE_DIR, exist_ok=True)

    os.makedirs(VEHICLE_CROP_DIR, exist_ok=True)
    os.makedirs(CROPPED_PLATE_DIR, exist_ok=True)

    os.makedirs(RECORDS_DIR, exist_ok=True)


def create_csv_if_not_exists():
    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_COLUMNS)

        print(f"[CSV] Created new CSV log: {CSV_LOG_PATH}")


def log_violation_to_csv(
    violation_type,
    vehicle_type,
    track_id="",
    plate_number="Unknown",
    plate_confidence=0.0,
    ocr_confidence=0.0,
    evidence_path="",
    vehicle_crop_path="",
    plate_crop_path=""
):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    row = [
        date_str,
        time_str,
        violation_type,
        vehicle_type,
        track_id if track_id is not None else "",
        plate_number,
        f"{plate_confidence:.2f}",
        f"{ocr_confidence:.2f}",
        evidence_path if evidence_path is not None else "",
        vehicle_crop_path if vehicle_crop_path is not None else "",
        plate_crop_path if plate_crop_path is not None else ""
    ]

    with open(CSV_LOG_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row)

    print(f"[CSV] Logged: {violation_type} | {vehicle_type} | Plate: {plate_number}")


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def save_evidence_image(frame, violation_type, vehicle_type="unknown", track_id=None):
    timestamp = get_timestamp()

    if violation_type == "Helmet Violation":
        folder = HELMET_EVIDENCE_DIR
        prefix = "helmet_violation"

    elif violation_type == "Red Light Violation":
        folder = RED_LIGHT_EVIDENCE_DIR
        prefix = "red_light_violation"

    elif violation_type == "Multiple Violation":
        folder = MULTIPLE_EVIDENCE_DIR
        prefix = "multiple_violation"

    else:
        folder = EVIDENCE_DIR
        prefix = "violation"

    if track_id is not None:
        file_name = f"{prefix}_{vehicle_type}_track{track_id}_{timestamp}.jpg"
    else:
        file_name = f"{prefix}_{vehicle_type}_{timestamp}.jpg"

    save_path = os.path.join(folder, file_name)

    success = cv2.imwrite(save_path, frame)

    if success:
        print(f"[SAVED] Evidence: {save_path}")
    else:
        print(f"[ERROR] Failed to save evidence: {save_path}")

    return save_path


def save_crop_image(image, folder, prefix, vehicle_type="unknown", track_id=None):
    timestamp = get_timestamp()

    if track_id is not None:
        file_name = f"{prefix}_{vehicle_type}_track{track_id}_{timestamp}.jpg"
    else:
        file_name = f"{prefix}_{vehicle_type}_{timestamp}.jpg"

    save_path = os.path.join(folder, file_name)

    success = cv2.imwrite(save_path, image)

    if success:
        print(f"[SAVED] Crop: {save_path}")
    else:
        print(f"[ERROR] Failed to save crop: {save_path}")

    return save_path

# ==============================
# UTILITY FUNCTIONS
# ==============================

def get_center(box):
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


def get_bottom_center(box):
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    bottom_y = y2
    return center_x, bottom_y


def get_top_center(box):
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    top_y = y1
    return center_x, top_y


def get_tracking_point(box):
    if ROAD_DIRECTION == "down":
        return get_bottom_center(box)

    elif ROAD_DIRECTION == "up":
        return get_bottom_center(box)

    return get_bottom_center(box)


def get_box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def expand_box(box, margin_x=0, margin_y=0):
    x1, y1, x2, y2 = box

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(FRAME_WIDTH, x2 + margin_x)
    y2 = min(FRAME_HEIGHT, y2 + margin_y)

    return x1, y1, x2, y2


def point_inside_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box

    return x1 <= px <= x2 and y1 <= py <= y2


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    inter_area = inter_width * inter_height

    box1_area = get_box_area(box1)
    box2_area = get_box_area(box2)

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


def remove_duplicate_detections(detections, iou_threshold=0.30):
    if len(detections) == 0:
        return []

    detections = sorted(
        detections,
        key=lambda x: (x["conf"], get_box_area(x["box"])),
        reverse=True
    )

    filtered = []

    for det in detections:
        keep = True

        for kept_det in filtered:
            if det["class"] != kept_det["class"]:
                continue

            iou = calculate_iou(det["box"], kept_det["box"])

            det_center = get_center(det["box"])
            kept_center = get_center(kept_det["box"])

            det_center_inside_kept = point_inside_box(det_center, kept_det["box"])
            kept_center_inside_det = point_inside_box(kept_center, det["box"])

            if iou > iou_threshold or det_center_inside_kept or kept_center_inside_det:
                keep = False
                break

        if keep:
            filtered.append(det)

    return filtered


def remove_helmet_conflicts(detections, iou_threshold=0.30):
    if len(detections) == 0:
        return []

    keep = [True] * len(detections)

    for i in range(len(detections)):
        if not keep[i]:
            continue

        for j in range(i + 1, len(detections)):
            if not keep[j]:
                continue

            class_i = detections[i]["class"]
            class_j = detections[j]["class"]

            if class_i in helmet_class_names and class_j in helmet_class_names and class_i != class_j:
                iou = calculate_iou(detections[i]["box"], detections[j]["box"])

                if iou > iou_threshold:
                    if detections[i]["conf"] >= detections[j]["conf"]:
                        keep[j] = False
                    else:
                        keep[i] = False

    filtered = []

    for index, det in enumerate(detections):
        if keep[index]:
            filtered.append(det)

    return filtered


def safe_crop(image, box, padding=0):
    img_h, img_w = image.shape[:2]

    x1, y1, x2, y2 = box

    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(img_w, int(x2) + padding)
    y2 = min(img_h, int(y2) + padding)

    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)

    crop = image[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)

def normalize_class_name(class_name):
    return str(class_name).lower().replace("_", " ").replace("-", " ").strip()


# ==============================
# OCR FUNCTIONS
# ==============================

def preprocess_plate_for_ocr(plate_image):
    if plate_image is None:
        return None

    plate_image = cv2.resize(
        plate_image,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    return gray


def clean_plate_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def read_plate_text(plate_image):
    processed_plate = preprocess_plate_for_ocr(plate_image)

    if processed_plate is None:
        return "Unknown", 0.0

    try:
        ocr_results = ocr_reader.readtext(processed_plate)
    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return "Unknown", 0.0

    if len(ocr_results) == 0:
        return "Unknown", 0.0

    best_text = ""
    best_conf = 0.0

    for result in ocr_results:
        bbox, text, conf = result

        cleaned_text = clean_plate_text(text)

        if len(cleaned_text) >= 3 and conf > best_conf:
            best_text = cleaned_text
            best_conf = conf

    if best_text == "" or best_conf < OCR_CONF_THRESHOLD:
        return "Unknown", best_conf

    return best_text, best_conf

# ==============================
# LICENSE PLATE DETECTION FUNCTIONS
# ==============================

def detect_and_save_plate(raw_frame, vehicle_box, violation_type, vehicle_type="unknown", track_id=None):
    vehicle_crop, vehicle_crop_box = safe_crop(
        raw_frame,
        vehicle_box,
        padding=15
    )

    if vehicle_crop is None:
        print("[PLATE] Vehicle crop failed.")
        return None, None, "Unknown", 0.0, 0.0

    vehicle_prefix = f"{violation_type}_vehicle"
    vehicle_crop_path = save_crop_image(
        vehicle_crop,
        VEHICLE_CROP_DIR,
        vehicle_prefix,
        vehicle_type,
        track_id
    )

    plate_results = model_plate(
        vehicle_crop,
        conf=CONF_THRESHOLD_PLATE,
        verbose=False
    )

    plate_detections = []

    for r in plate_results:
        for box in r.boxes:
            px1, py1, px2, py2 = box.xyxy[0]
            px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model_plate.names[cls_id]

            plate_detections.append({
                "class": str(class_name).lower(),
                "box": (px1, py1, px2, py2),
                "conf": conf
            })

    if len(plate_detections) == 0:
        print("[PLATE] No plate detected.")
        return vehicle_crop_path, None, "Unknown", 0.0, 0.0

    best_plate = max(plate_detections, key=lambda x: x["conf"])
    plate_box = best_plate["box"]
    plate_conf = best_plate["conf"]

    plate_crop, plate_crop_box = safe_crop(
        vehicle_crop,
        plate_box,
        padding=8
    )

    if plate_crop is None:
        print("[PLATE] Plate crop failed.")
        return vehicle_crop_path, None, "Unknown", plate_conf, 0.0

    plate_prefix = f"{violation_type}_plate"
    plate_crop_path = save_crop_image(
        plate_crop,
        CROPPED_PLATE_DIR,
        plate_prefix,
        vehicle_type,
        track_id
    )

    print(f"[PLATE] Plate detected. Confidence: {plate_conf:.2f}")

    plate_text, ocr_conf = read_plate_text(plate_crop)

    print(f"[OCR] Plate Number: {plate_text} | OCR Confidence: {ocr_conf:.2f}")

    return vehicle_crop_path, plate_crop_path, plate_text, plate_conf, ocr_conf

# ==============================
# HELMET VIOLATION FUNCTIONS
# ==============================

def is_head_inside_person(head_box, person_box):
    head_center = get_center(head_box)

    expanded_person_box = expand_box(
        person_box,
        margin_x=PERSON_MARGIN,
        margin_y=PERSON_MARGIN
    )

    return point_inside_box(head_center, expanded_person_box)


def is_person_associated_with_motorcycle(person_box, motorcycle_box):
    px1, py1, px2, py2 = person_box
    mx1, my1, mx2, my2 = motorcycle_box

    person_center_x = int((px1 + px2) / 2)
    person_bottom_y = py2

    expanded_motorcycle_x1 = mx1 - MOTORCYCLE_X_MARGIN
    expanded_motorcycle_x2 = mx2 + MOTORCYCLE_X_MARGIN

    expanded_motorcycle_y1 = my1 - MOTORCYCLE_TOP_MARGIN
    expanded_motorcycle_y2 = my2 + MOTORCYCLE_BOTTOM_MARGIN

    x_match = expanded_motorcycle_x1 <= person_center_x <= expanded_motorcycle_x2
    y_match = expanded_motorcycle_y1 <= person_bottom_y <= expanded_motorcycle_y2

    return x_match and y_match


def check_helmet_violation(without_helmet_boxes, person_boxes, motorcycle_boxes):
    helmet_violations = []

    for head_box in without_helmet_boxes:
        matched_person = None
        matched_motorcycle = None

        for person_box in person_boxes:
            if is_head_inside_person(head_box, person_box):
                matched_person = person_box
                break

        if matched_person is not None:
            for motorcycle_box in motorcycle_boxes:
                if is_person_associated_with_motorcycle(matched_person, motorcycle_box):
                    matched_motorcycle = motorcycle_box
                    break

        if matched_person is not None and matched_motorcycle is not None:
            helmet_violations.append({
                "head_box": head_box,
                "person_box": matched_person,
                "motorcycle_box": matched_motorcycle,
                "violation_type": "Without Helmet"
            })

    return helmet_violations

# ==============================
# TRAFFIC LIGHT FUNCTIONS
# ==============================

def get_current_traffic_light(traffic_detections):
    traffic_lights = []

    for det in traffic_detections:
        if det["class"] in traffic_light_class_names:
            traffic_lights.append(det)

    if len(traffic_lights) == 0:
        return "unknown", 0.0

    best_light = max(traffic_lights, key=lambda x: x["conf"])

    return best_light["class"], best_light["conf"]

# ==============================
# VEHICLE TRACKING FUNCTIONS
# ==============================

def distance_between_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def update_vehicle_tracks(vehicle_detections, tracks, next_id):
    for track_id in tracks:
        tracks[track_id]["missing"] += 1

    matched_track_ids = set()

    for det in vehicle_detections:
        if det["class"] not in red_light_vehicle_classes:
            continue

        box = det["box"]
        center = get_tracking_point(box)

        best_track_id = None
        best_distance = TRACK_MAX_DISTANCE

        for track_id, track in tracks.items():
            if track_id in matched_track_ids:
                continue

            distance = distance_between_points(center, track["center"])

            if distance < best_distance:
                best_distance = distance
                best_track_id = track_id

        if best_track_id is not None:
            tracks[best_track_id]["prev_center"] = tracks[best_track_id]["center"]
            tracks[best_track_id]["center"] = center
            tracks[best_track_id]["box"] = box
            tracks[best_track_id]["class"] = det["class"]
            tracks[best_track_id]["conf"] = det["conf"]
            tracks[best_track_id]["missing"] = 0

            matched_track_ids.add(best_track_id)
            det["track_id"] = best_track_id

        else:
            tracks[next_id] = {
                "prev_center": center,
                "center": center,
                "box": box,
                "class": det["class"],
                "conf": det["conf"],
                "missing": 0,
                "red_violation_done": False
            }

            det["track_id"] = next_id
            matched_track_ids.add(next_id)

            next_id += 1

    lost_ids = []

    for track_id, track in tracks.items():
        if track["missing"] > TRACK_MAX_MISSING:
            lost_ids.append(track_id)

    for track_id in lost_ids:
        del tracks[track_id]

    return tracks, next_id


def is_crossing_stop_line(prev_y, current_y):
    if ROAD_DIRECTION == "down":
        return prev_y < STOP_LINE_Y <= current_y

    elif ROAD_DIRECTION == "up":
        return prev_y > STOP_LINE_Y >= current_y

    return False


def check_red_light_violation(current_light, tracks):
    red_light_violations = []

    if current_light != "red light":
        for track_id in tracks:
            tracks[track_id]["red_violation_done"] = False

        return red_light_violations

    for track_id, track in tracks.items():
        if track["class"] not in red_light_vehicle_classes:
            continue

        if track["red_violation_done"]:
            continue

        prev_y = track["prev_center"][1]
        current_y = track["center"][1]

        if is_crossing_stop_line(prev_y, current_y):
            track["red_violation_done"] = True

            red_light_violations.append({
                "track_id": track_id,
                "vehicle_type": track["class"],
                "vehicle_box": track["box"],
                "violation_type": "Red Light Violation"
            })

    return red_light_violations

# ==============================
# COLOR FUNCTIONS
# ==============================

def get_helmet_color(class_name):
    class_name = class_name.lower()

    if class_name == "with helmet":
        return (0, 255, 0)

    elif class_name == "without helmet":
        return (0, 0, 255)

    else:
        return (255, 255, 255)


def get_traffic_color(class_name):
    class_name = normalize_class_name(class_name)

    if class_name == "red light":
        return (0, 0, 255)

    elif class_name == "yellow light":
        return (0, 255, 255)

    elif class_name == "green light":
        return (0, 255, 0)

    else:
        return (255, 255, 255)


def get_vehicle_color(class_name):
    class_name = class_name.lower()

    if class_name == "person":
        return (255, 0, 255)

    elif class_name in ["motorcycle", "motorbike"]:
        return (255, 255, 0)

    elif class_name == "car":
        return (255, 0, 0)

    elif class_name == "truck":
        return (0, 165, 255)

    elif class_name == "bus":
        return (128, 0, 255)

    else:
        return (255, 255, 255)

# ==============================
# DRAWING FUNCTIONS
# ==============================

def draw_box(frame, box, label, color, thickness=1):
    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    text_size, _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2
    )

    text_w, text_h = text_size

    if y1 - text_h - 10 < 0:
        label_y1 = y1
        label_y2 = y1 + text_h + 10
        text_y = y1 + text_h + 5
    else:
        label_y1 = y1 - text_h - 10
        label_y2 = y1
        text_y = y1 - 5

    cv2.rectangle(
        frame,
        (x1, label_y1),
        (x1 + text_w + 10, label_y2),
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 5, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2
    )


def draw_stop_line(frame):
    cv2.line(
        frame,
        (0, STOP_LINE_Y),
        (FRAME_WIDTH, STOP_LINE_Y),
        (0, 255, 255),
        3
    )

    cv2.putText(
        frame,
        "Virtual Stop Line",
        (20, STOP_LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


def draw_helmet_violation_warning(frame, violation):
    head_box = violation["head_box"]
    person_box = violation["person_box"]
    motorcycle_box = violation["motorcycle_box"]

    hx1, hy1, hx2, hy2 = head_box
    px1, py1, px2, py2 = person_box
    mx1, my1, mx2, my2 = motorcycle_box

    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 4)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 255), 3)

    head_center = get_center(head_box)
    motorcycle_center = get_center(motorcycle_box)
    cv2.line(frame, head_center, motorcycle_center, (0, 0, 255), 2)


def draw_red_light_violation_warning(frame, violation):
    box = violation["vehicle_box"]
    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

    label = "RLV"

    cv2.rectangle(
        frame,
        (x1, max(0, y1 - 25)),
        (x1 + 65, y1),
        (0, 0, 255),
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 10, y1 - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


def draw_status_panel(frame, fps, current_light, light_conf, helmet_count, redlight_count):
    frame_h, frame_w = frame.shape[:2]

    # ==============================
    # PANEL POSITION - TOP RIGHT
    # ==============================
    # panel_width = 330
    # panel_height = 95
    # margin = 15
    #
    # x1 = frame_w - panel_width - margin
    # y1 = margin
    # x2 = frame_w - margin
    # y2 = y1 + panel_height

    # ==============================
    # PANEL POSITION - TOP LEFT
    # ==============================
    panel_width = 330
    panel_height = 95
    margin = 15

    x1 = margin
    y1 = margin
    x2 = x1 + panel_width
    y2 = y1 + panel_height


    # Safety kalau frame kecil
    x1 = max(10, x1)
    y1 = max(10, y1)

    # ==============================
    # PANEL BACKGROUND
    # ==============================
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (20, 20, 20),
        -1
    )

    # Transparent effect
    alpha = 0.78
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Panel border
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    # ==============================
    # DETERMINE VIOLATION TEXT
    # ==============================
    if helmet_count > 0 and redlight_count > 0:
        violation_text = "MULTIPLE VIOLATION"
        violation_color = (0, 0, 255)

    elif helmet_count > 0:
        violation_text = "WITHOUT HELMET DETECTED"
        violation_color = (0, 0, 255)

    elif redlight_count > 0:
        violation_text = "RED LIGHT VIOLATION"
        violation_color = (0, 0, 255)

    else:
        violation_text = "NO VIOLATION"
        violation_color = (0, 255, 0)

    # ==============================
    # TEXT DISPLAY
    # ==============================

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (x1 + 12, y1 + 25),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    # System title
    cv2.putText(
        frame,
        "Traffic Violation Detection System",
        (x1 + 12, y1 + 55),
        cv2.FONT_HERSHEY_TRIPLEX,
        0.50,
        (255, 255, 255),
        1
    )

    # Violation label
    cv2.putText(
        frame,
        "Violation:",
        (x1 + 12, y1 + 82),
        cv2.FONT_HERSHEY_TRIPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        violation_text,
        (x1 + 105, y1 + 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        violation_color,
        2
    )




    y_alert = 110

    if redlight_count > 0:
        cv2.rectangle(
            frame,
            (20, y_alert),
            (295, y_alert + 35),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            "RED LIGHT VIOLATION",
            (30, y_alert + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    elif helmet_count > 0:
        cv2.rectangle(
            frame,
            (20, y_alert),
            (265, y_alert + 35),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            "HELMET VIOLATION",
            (30, y_alert + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

# ==============================
# MAIN LOOP
# ==============================

create_output_folders()
create_csv_if_not_exists()
print("Output folders and CSV log ready.")

cv2.namedWindow("FYP Traffic Violation Detection - Separated Models", cv2.WINDOW_NORMAL)

prev_time = 0

while True:
    success, frame = cap.read()

    if not success:
        print("Video ended or frame cannot be read.")
        break

    if USE_ORIGINAL_VIDEO_SIZE:
        FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]
    else:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    raw_frame = frame.copy()

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    without_helmet_boxes = []
    with_helmet_boxes = []
    person_boxes = []
    motorcycle_boxes = []
    vehicle_boxes = []

    helmet_detections_raw = []
    traffic_detections_raw = []
    vehicle_detections_raw = []

    # ==============================
    # RUN HELMET MODEL
    # ==============================

    helmet_results = model_helmet(
        frame,
        stream=True,
        conf=CONF_THRESHOLD_HELMET
    )

    for r in helmet_results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model_helmet.names[cls_id]
            class_name_lower = normalize_class_name(class_name)

            current_box = (x1, y1, x2, y2)

            if class_name_lower not in helmet_class_names:
                continue

            helmet_detections_raw.append({
                "class": class_name_lower,
                "display_class": class_name,
                "box": current_box,
                "conf": conf
            })

    helmet_detections = remove_duplicate_detections(
        helmet_detections_raw,
        iou_threshold=HELMET_DUPLICATE_IOU
    )

    helmet_detections = remove_helmet_conflicts(
        helmet_detections,
        iou_threshold=HELMET_CONFLICT_IOU
    )

    for det in helmet_detections:
        class_name_lower = det["class"]
        class_name = det["display_class"]
        current_box = det["box"]
        conf = det["conf"]

        if class_name_lower == "without helmet":
            without_helmet_boxes.append(current_box)

        elif class_name_lower == "with helmet":
            with_helmet_boxes.append(current_box)

        if SHOW_NORMAL_DETECTIONS:
            color = get_helmet_color(class_name)
            label = f"{class_name} {conf:.2f}"
            draw_box(frame, current_box, label, color, 2)

    # ==============================
    # RUN TRAFFIC LIGHT MODEL
    # ==============================

    traffic_results = model_traffic(
        frame,
        stream=True,
        conf=CONF_THRESHOLD_TRAFFIC_LIGHT
    )

    for r in traffic_results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model_traffic.names[cls_id]
            class_name_lower = normalize_class_name(class_name)

            current_box = (x1, y1, x2, y2)

            if class_name_lower not in traffic_light_class_names:
                continue

            traffic_detections_raw.append({
                "class": class_name_lower,
                "display_class": class_name,
                "box": current_box,
                "conf": conf
            })

    traffic_detections = remove_duplicate_detections(
        traffic_detections_raw,
        iou_threshold=TRAFFIC_DUPLICATE_IOU
    )

    detected_light, detected_light_conf = get_current_traffic_light(traffic_detections)

    if detected_light != "unknown":
        last_known_light = detected_light
        last_known_light_conf = detected_light_conf
        traffic_light_missing_counter = 0
    else:
        traffic_light_missing_counter += 1

    if traffic_light_missing_counter <= TRAFFIC_LIGHT_MEMORY_FRAMES:
        current_light = last_known_light
        current_light_conf = last_known_light_conf
    else:
        current_light = "unknown"
        current_light_conf = 0.0

    for det in traffic_detections:
        class_name = det["display_class"]
        current_box = det["box"]
        conf = det["conf"]

        if SHOW_NORMAL_DETECTIONS:
            color = get_traffic_color(class_name)
            label = f"{class_name} {conf:.2f}"
            draw_box(frame, current_box, label, color, 2)

    # ==============================
    # RUN VEHICLE / PERSON MODEL
    # ==============================

    vehicle_results = model_vehicle(
        frame,
        stream=True,
        conf=CONF_THRESHOLD_VEHICLE
    )

    for r in vehicle_results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model_vehicle.names[cls_id]
            class_name_lower = class_name.lower()

            if class_name_lower not in vehicle_related_classes:
                continue

            current_box = (x1, y1, x2, y2)

            vehicle_detections_raw.append({
                "class": class_name_lower,
                "display_class": class_name,
                "box": current_box,
                "conf": conf
            })

    vehicle_detections = remove_duplicate_detections(
        vehicle_detections_raw,
        iou_threshold=VEHICLE_DUPLICATE_IOU
    )

    for det in vehicle_detections:
        class_name_lower = det["class"]
        class_name = det["display_class"]
        current_box = det["box"]
        conf = det["conf"]

        if class_name_lower == "person":
            person_boxes.append(current_box)

        elif class_name_lower in motorcycle_class_names:
            motorcycle_boxes.append(current_box)
            vehicle_boxes.append(det)

        elif class_name_lower in ["car", "truck", "bus"]:
            vehicle_boxes.append(det)

        if SHOW_NORMAL_DETECTIONS:
            color = get_vehicle_color(class_name)
            label = f"{class_name} {conf:.2f}"
            draw_box(frame, current_box, label, color, 2)

    # ==============================
    # UPDATE VEHICLE TRACKING
    # ==============================

    vehicle_tracks, next_track_id = update_vehicle_tracks(
        vehicle_boxes,
        vehicle_tracks,
        next_track_id
    )

    # ==============================
    # HELMET VIOLATION LOGIC
    # ==============================

    raw_helmet_violations = check_helmet_violation(
        without_helmet_boxes,
        person_boxes,
        motorcycle_boxes
    )

    if len(raw_helmet_violations) > 0:
        helmet_violation_counter += 1
        helmet_clear_counter = 0

        if helmet_violation_counter >= HELMET_CONFIRM_FRAMES:
            helmet_violation_active = True
            last_confirmed_helmet_violations = raw_helmet_violations

    else:
        helmet_violation_counter = max(0, helmet_violation_counter - 1)

        if helmet_violation_active:
            helmet_clear_counter += 1

            if helmet_clear_counter >= HELMET_CLEAR_FRAMES:
                helmet_violation_active = False
                helmet_clear_counter = 0
                last_confirmed_helmet_violations = []

    if helmet_violation_active:
        confirmed_helmet_violations = last_confirmed_helmet_violations
    else:
        confirmed_helmet_violations = []

    # ==============================
    # RED LIGHT VIOLATION LOGIC
    # ==============================

    new_red_light_violations = check_red_light_violation(
        current_light,
        vehicle_tracks
    )

    for violation in new_red_light_violations:
        violation["frames_left"] = RED_LIGHT_DISPLAY_FRAMES
        active_red_light_events.append(violation)

    updated_events = []

    for event in active_red_light_events:
        if event["frames_left"] > 0:
            updated_events.append(event)
            event["frames_left"] -= 1

    active_red_light_events = updated_events

    # ==============================
    # DRAW VIOLATIONS
    # ==============================

    if SHOW_STOP_LINE:
        draw_stop_line(frame)

    for violation in confirmed_helmet_violations:
        draw_helmet_violation_warning(frame, violation)

    for violation in active_red_light_events:
        draw_red_light_violation_warning(frame, violation)

    # ==============================
    # SAVE EVIDENCE + PLATE OCR + CSV
    # ==============================

    current_save_time = time.time()

    # Helmet violation
    if len(confirmed_helmet_violations) > 0:
        if not helmet_evidence_saved and (current_save_time - last_helmet_save_time) >= HELMET_SAVE_COOLDOWN:
            evidence_path = save_evidence_image(
                frame,
                violation_type="Helmet Violation",
                vehicle_type="motorcycle"
            )

            first_violation = confirmed_helmet_violations[0]

            vehicle_crop_path, plate_crop_path, plate_text, plate_conf, ocr_conf = detect_and_save_plate(
                raw_frame,
                first_violation["motorcycle_box"],
                violation_type="helmet",
                vehicle_type="motorcycle"
            )

            log_violation_to_csv(
                violation_type="Helmet Violation",
                vehicle_type="motorcycle",
                track_id="",
                plate_number=plate_text,
                plate_confidence=plate_conf,
                ocr_confidence=ocr_conf,
                evidence_path=evidence_path,
                vehicle_crop_path=vehicle_crop_path,
                plate_crop_path=plate_crop_path
            )

            helmet_evidence_saved = True
            last_helmet_save_time = current_save_time

    else:
        helmet_evidence_saved = False

    # Red light violation
    if len(new_red_light_violations) > 0:
        if (current_save_time - last_red_light_save_time) >= RED_LIGHT_SAVE_COOLDOWN:
            for violation in new_red_light_violations:
                evidence_path = save_evidence_image(
                    frame,
                    violation_type="Red Light Violation",
                    vehicle_type=violation["vehicle_type"],
                    track_id=violation["track_id"]
                )

                vehicle_crop_path, plate_crop_path, plate_text, plate_conf, ocr_conf = detect_and_save_plate(
                    raw_frame,
                    violation["vehicle_box"],
                    violation_type="red_light",
                    vehicle_type=violation["vehicle_type"],
                    track_id=violation["track_id"]
                )

                log_violation_to_csv(
                    violation_type="Red Light Violation",
                    vehicle_type=violation["vehicle_type"],
                    track_id=violation["track_id"],
                    plate_number=plate_text,
                    plate_confidence=plate_conf,
                    ocr_confidence=ocr_conf,
                    evidence_path=evidence_path,
                    vehicle_crop_path=vehicle_crop_path,
                    plate_crop_path=plate_crop_path
                )

            last_red_light_save_time = current_save_time

    # ==============================
    # DISPLAY INFO
    # ==============================

    draw_status_panel(
        frame,
        fps,
        current_light,
        current_light_conf,
        len(confirmed_helmet_violations),
        len(active_red_light_events)
    )

    cv2.imshow("FYP Traffic Violation Detection - Separated Models", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

# ==============================
# RELEASE
# ==============================

cap.release()
cv2.destroyAllWindows()
print("Program ended.")