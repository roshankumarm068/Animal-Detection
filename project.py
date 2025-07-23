import cv2
import torch
import os
import time
import threading
from datetime import datetime
import customtkinter as ctk
from tkinter import messagebox

# ================================
# CONFIGURATION SETTINGS
# ================================
VIDEO_SOURCE = 0
CONFIDENCE_THRESHOLD = 0.4
OUTPUT_DIR = 'output'
LOG_FILE = os.path.join(OUTPUT_DIR, 'detection_log.txt')
SAVE_VIDEO = True
SOUND_ALERT = True

ANIMAL_CLASSES = [
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'goat', 'deer', 'monkey', 'rabbit', 'mouse', 'fox', 'wolf', 'lion', 'tiger', 'leopard',
    'otter', 'kangaroo', 'panda', 'crocodile', 'hippopotamus', 'rhinoceros', 'camel', 'donkey',
    'squirrel', 'rat', 'bat', 'hedgehog', 'badger', 'ferret', 'weasel', 'mole', 'shrew', 'porcupine',
    'chicken', 'duck', 'goose', 'turkey', 'owl', 'eagle', 'falcon', 'parrot', 'penguin', 'peacock',
    'dolphin', 'whale', 'shark', 'seal', 'sea_lion', 'crab', 'lobster', 'octopus', 'fish',
    'frog', 'toad', 'turtle', 'snake', 'lizard', 'crocodile', 'alligator', 'newt', 'salamander',
    'bee', 'ant', 'butterfly', 'moth', 'dragonfly', 'grasshopper', 'ladybug', 'spider', 'scorpion',
    'snail', 'slug', 'worm', 'centipede', 'millipede'
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Helper for UI-safe status updates
def set_status(label, text, color="#4ade80"):
    label.configure(text=text, text_color=color)

stop_detection_flag = False
model = None

# Add threading lock to prevent race condition
thread_lock = threading.Lock()

# Twilio SMS notification setup

# Twilio REST API credentials and settings
TWILIO_ACCOUNT_SID = 'ACd96048876e15f858db5026a3ee353792'
TWILIO_AUTH_TOKEN = 'your_auth_token'  # Replace with your actual Auth Token
TWILIO_MESSAGING_SERVICE_SID = 'MG194b73c992ec8ea26803b1aabdfebe32'
ALERT_PHONE_NUMBER = '+9779816738058'

import requests
from requests.auth import HTTPBasicAuth

def send_sms_alert(message):
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    data = {
        'To': ALERT_PHONE_NUMBER,
        'MessagingServiceSid': TWILIO_MESSAGING_SERVICE_SID,
        'Body': message
    }
    try:
        response = requests.post(url, data=data, auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        if response.status_code == 201:
            print(f"SMS sent: {message}")
        else:
            print(f"Failed to send SMS: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Failed to send SMS: {e}")


def load_yolo_model():
    global model
    if model is None:
        print("⏳ Loading YOLOv5 model...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to("cpu")
        print("✅ YOLOv5 model loaded.")
    return model


def play_sound():
    try:
        import winsound
        winsound.Beep(1000, 500)
    except ImportError:
        print('\a', end='')


def log_detection(label, confidence, timestamp):
    with open(LOG_FILE, 'a') as f:
        f.write(f"{timestamp} - Detected: {label} ({confidence:.2f})\n")


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def draw_alert(frame):
    cv2.putText(frame, "🚨 ALERT: Animal Detected!", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


def run_detection_ui_safe(status_label, stats_label, start_btn, stop_btn, video_source, conf_thresh):
    global stop_detection_flag
    model = load_yolo_model()
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        set_status(status_label, "❌ Error: Could not open video source", "#dc2626")
        messagebox.showerror("Error", "Could not open video source.")
        start_btn.configure(state="normal")
        stop_btn.configure(state="disabled")
        return

    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, 'output.avi'), fourcc, 20.0, (640, 480))

    frame_count = 0
    animal_frame_count = 0
    start_time = time.time()

    set_status(status_label, "🎥 Detection started. Press 'q' to quit.", "#4ade80")
    stats_label.configure(text="Frames: 0 | Animal Frames: 0")

    import PIL.Image, PIL.ImageTk
    # Use video/info widgets from main window
    video_label = launch_ui.video_frame
    info_box = launch_ui.info_frame


    detected_animals = []
    last_animal_details = None  # Store last detected animal info

    def process_frame():
        nonlocal frame_count, animal_frame_count, detected_animals, last_animal_details
        with thread_lock:
            if stop_detection_flag:
                cap.release()
                if out:
                    out.release()
                set_status(status_label, f"✅ Detection stopped. Frames: {frame_count}, Animal Frames: {animal_frame_count}", "#4ade80")
                start_btn.configure(state="normal")
                stop_btn.configure(state="disabled")
                return

        ret, frame = cap.read()
        if not ret:
            set_status(status_label, "❌ End of video or failed to read frame.", "#dc2626")
            cap.release()
            if out:
                out.release()
            start_btn.configure(state="normal")
            stop_btn.configure(state="disabled")
            return

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)
        detections = results.pandas().xyxy[0]

        animal_detected = False
        animal_details = []
        for _, row in detections.iterrows():
            label = row['name']
            confidence = float(row['confidence'])
            if label in ANIMAL_CLASSES and confidence > conf_thresh:
                animal_detected = True
                x1, y1 = int(row['xmin']), int(row['ymin'])
                x2, y2 = int(row['xmax']), int(row['ymax'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_detection(label, confidence, timestamp)
                animal_details.append(f"{timestamp}: {label} ({confidence:.2f})")
                last_animal_details = f"{label} ({confidence:.2f}) detected at {timestamp}"
                # Send SMS notification (only once per detection loop)
                if animal_detected and len(animal_details) == 1:
                    sms_message = f"ALERT: {label} detected at {timestamp} (confidence: {confidence:.2f})"
                    threading.Thread(target=send_sms_alert, args=(sms_message,), daemon=True).start()

        if animal_detected:
            animal_frame_count += 1
            draw_alert(frame)
            set_status(status_label, "🚨 Animal detected!", "#dc2626")
            if SOUND_ALERT:
                threading.Thread(target=play_sound).start()
            detected_animals.extend(animal_details)
        else:
            set_status(status_label, "🐾 Scanning for animals...", "#4ade80")

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        draw_fps(frame, fps)
        stats_label.configure(text=f"Frames: {frame_count} | Animal Frames: {animal_frame_count}")

        # Show frame in Tkinter window
        img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk  # persistent reference
        video_label.configure(image=imgtk, text="")

        # Show animal details in info box
        info_box.configure(state="normal")
        info_box.delete("0.0", "end")
        if last_animal_details:
            info_box.insert("end", f"Last Detected Animal:\n{last_animal_details}")
        else:
            info_box.insert("end", "No animals detected yet.")
        info_box.configure(state="disabled")

        if SAVE_VIDEO and out:
            out.write(frame)

        # Schedule next frame update for smooth video (about 30 FPS)
        video_label.after(33, process_frame)

    process_frame()


def start_detection(status_label, stats_label, start_btn, stop_btn, video_source_var, conf_thresh_var):
    global stop_detection_flag
    stop_detection_flag = False
    start_btn.configure(state="disabled")
    stop_btn.configure(state="normal")
    video_source = int(video_source_var.get()) if video_source_var.get().isdigit() else video_source_var.get()
    conf_thresh = float(conf_thresh_var.get())
    threading.Thread(target=run_detection_ui_safe, args=(status_label, stats_label, start_btn, stop_btn, video_source, conf_thresh), daemon=True).start()


def stop_detection(status_label, start_btn, stop_btn):
    global stop_detection_flag
    with thread_lock:
        stop_detection_flag = True
    messagebox.showinfo("Stopping", "Detection will stop shortly. Close video window or press 'q'.")
    set_status(status_label, "🛑 Stopping detection...", "#dc2626")
    start_btn.configure(state="normal")
    stop_btn.configure(state="disabled")


def launch_ui():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("🐾 Animal Detection System")
    root.geometry("900x520")
    root.resizable(False, False)
    main_frame = ctk.CTkFrame(master=root)
    main_frame.pack(padx=10, pady=10, fill="both", expand=True)

    # Left panel: controls
    left_panel = ctk.CTkFrame(main_frame, width=260)
    left_panel.pack(side="left", fill="y", expand=False)

    title_label = ctk.CTkLabel(left_panel, text="🐾 Animal Detection using YOLOv5", font=ctk.CTkFont(size=22, weight="bold"))
    title_label.pack(pady=(20, 10))
    subtitle = ctk.CTkLabel(left_panel, text="Real-time animal identification", font=ctk.CTkFont(size=14), text_color="gray")
    subtitle.pack(pady=(0, 10))
    status_label = ctk.CTkLabel(left_panel, text="🟢 Ready to start detection", font=ctk.CTkFont(size=16), text_color="#4ade80")
    status_label.pack(pady=(0, 10))
    stats_label = ctk.CTkLabel(left_panel, text="Frames: 0 | Animal Frames: 0", font=ctk.CTkFont(size=12), text_color="gray")
    stats_label.pack(pady=(0, 10))

    video_source_var = ctk.StringVar(value=str(VIDEO_SOURCE))
    video_source_label = ctk.CTkLabel(left_panel, text="Video Source (0=Webcam, or file path):", font=ctk.CTkFont(size=12))
    video_source_label.pack()
    video_source_entry = ctk.CTkEntry(left_panel, textvariable=video_source_var, width=180)
    video_source_entry.pack(pady=(0, 10))

    conf_thresh_var = ctk.StringVar(value=str(CONFIDENCE_THRESHOLD))
    conf_thresh_label = ctk.CTkLabel(left_panel, text="Confidence Threshold (0.0 - 1.0):", font=ctk.CTkFont(size=12))
    conf_thresh_label.pack()
    conf_thresh_entry = ctk.CTkEntry(left_panel, textvariable=conf_thresh_var, width=100)
    conf_thresh_entry.pack(pady=(0, 10))

    btn_container = ctk.CTkFrame(left_panel)
    btn_container.pack(pady=10, fill="x", expand=False)
    start_btn = ctk.CTkButton(btn_container, text="▶ Start Detection", font=ctk.CTkFont(size=16, weight="bold"), fg_color="#2563eb", hover_color="#1e40af", command=lambda: start_detection(status_label, stats_label, start_btn, stop_btn, video_source_var, conf_thresh_var))
    start_btn.pack(side="left", expand=True, fill="x", padx=5, ipady=8)
    stop_btn = ctk.CTkButton(btn_container, text="⏹ Stop Detection", font=ctk.CTkFont(size=16, weight="bold"), fg_color="#dc2626", hover_color="#991b1b", state="disabled", command=lambda: stop_detection(status_label, start_btn, stop_btn))
    stop_btn.pack(side="left", expand=True, fill="x", padx=5, ipady=8)
    exit_btn = ctk.CTkButton(left_panel, text="❌ Exit", font=ctk.CTkFont(size=14), fg_color="#6b7280", hover_color="#374151", command=root.destroy)
    exit_btn.pack(pady=(20, 10), ipadx=8, ipady=4)
    credits = ctk.CTkLabel(left_panel, text="Made by Prachit, Roshan, and Rahul", text_color="gray", font=ctk.CTkFont(size=10))
    credits.pack(side="bottom", pady=5)

    # Right panel: video and info
    right_panel = ctk.CTkFrame(main_frame)
    right_panel.pack(side="left", fill="both", expand=True)
    video_frame = ctk.CTkLabel(right_panel, text="Video loading...", width=640, height=480)
    video_frame.place(x=10, y=10)
    info_frame = ctk.CTkTextbox(right_panel, width=220, height=480)
    info_frame.place(x=660, y=10)

    # Store references for detection function
    launch_ui.video_frame = video_frame
    launch_ui.info_frame = info_frame
    launch_ui.status_label = status_label
    launch_ui.stats_label = stats_label
    launch_ui.start_btn = start_btn
    launch_ui.stop_btn = stop_btn
    launch_ui.video_source_var = video_source_var
    launch_ui.conf_thresh_var = conf_thresh_var

    root.mainloop()


if __name__ == "__main__":
    launch_ui()