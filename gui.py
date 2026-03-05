import os
import cv2
import csv
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
from datetime import datetime
from PIL import Image, ImageTk
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.filters import sobel
from skimage.measure import shannon_entropy
import threading
import time
from queue import Queue

# ------------------------------
# Global variables and Model
# ------------------------------
model = YOLO('best.onnx')

img_path = ""
pred = ""
conf = 0
last_lime_path = ""
last_removed_bg_path = ""
major_feature = ""
feature_importance = {}
matplotlib.use("Agg")

# ------------------------------
# Detection
# ------------------------------
def detect_mango(image_path):
    global pred, conf
    results = model(image_path)
    boxes = results[0].boxes
    if boxes and boxes.shape[0] > 0:
        pred = results[0].names[int(boxes.cls[0])]
        conf = float(boxes.conf[0]) * 100
    else:
        pred = "No object"
        conf = 0
    return pred, conf, results[0].plot()

def capture_from_camera():
    """Capture a single frame from the Pi camera and return path."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not access the camera.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        messagebox.showerror("Camera Error", "Failed to capture image.")
        return None

    os.makedirs("camera_captures", exist_ok=True)
    img_path = os.path.join("camera_captures", "captured_image.jpg")
    cv2.imwrite(img_path, frame)

    return img_path

def show_image(image_path, label_widget, max_size=(300, 300)):
    img = Image.open(image_path)
    img.thumbnail(max_size)
    tk_img = ImageTk.PhotoImage(img)
    label_widget.config(image=tk_img)
    label_widget.image = tk_img

# ------------------------------
# Classifier function for LIME
# ------------------------------
def model_fn(images):
    results_arr = []
    for img in images:
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = model(img_bgr, verbose=False)
        scores = np.zeros(len(model.names))
        for r in result:
            for cls_id, c in zip(r.boxes.cls, r.boxes.conf):
                scores[int(cls_id.item())] += c.item()
        results_arr.append(scores)
    return np.array(results_arr)

# ------------------------------
# Background Removal
# ------------------------------
def remove_background_with_yolo(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.zeros(image.shape[:2], np.uint8)

    results = model(image_path)
    boxes = results[0].boxes
    if not boxes or boxes.shape[0] == 0:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA), None

    box = boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    rect = (x1, y1, x2 - x1, y2 - y1)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image_rgb, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    image_fg = image_rgb * mask2[:, :, np.newaxis]

    alpha = mask2 * 255
    r, g, b = cv2.split(image_fg)
    rgba = cv2.merge((r, g, b, alpha))

    os.makedirs("background_removed_outputs", exist_ok=True)
    base_name = os.path.basename(image_path)
    output_path = os.path.join("background_removed_outputs", f"bg_removed_{base_name}")
    cv2.imwrite(output_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    return rgba, output_path

# ------------------------------
# Crop helper
# ------------------------------
def crop_image_to_yolo_box(image_path):
    results = model(image_path)
    boxes = results[0].boxes
    if not boxes or boxes.shape[0] == 0:
        return None
    x1, y1, x2, y2 = map(int, boxes[0].xyxy[0].cpu().numpy())
    img = cv2.imread(image_path)
    cropped = img[y1:y2, x1:x2]
    return cropped

def save_temp_cropped_image(cropped_img):
    path = "crop_lime/temp_cropped_img.png"
    os.makedirs("crop_lime", exist_ok=True)
    cv2.imwrite(path, cropped_img)
    return path

# ------------------------------
# Feature classifier + Bar chart
# ------------------------------
def classify_segments(mask, segments, image_rgb, lime_weights=None):
    h, w = mask.shape[:2]
    segment_ids = np.unique(segments)

    feature_scores = {
        "No defects": 0,
        "Smooth Texture": 0,
        "Minor defects": 0,
        "Blemished": 0,
        "Rough Texture": 0,
        "Dents": 0,
        "Molded": 0,
        "Damaged": 0,
        "Punctured": 0
    }

    
    image_gray = np.mean(image_rgb, axis=2)
    sobel_map = sobel(image_gray)

    for seg_id in segment_ids:
        if seg_id == 0:
            continue

        seg_mask = (segments == seg_id)

      
        if lime_weights is not None:
            importance = lime_weights.get(seg_id, 0.0)
        else:
            importance = np.sum(mask[seg_mask])

        if importance == 0:
            continue

        gray_region = image_gray[seg_mask]
        rgb_region = image_rgb[seg_mask]

        std_dev = np.std(gray_region)
        entropy = shannon_entropy(gray_region)
        edge_density = np.mean(sobel_map[seg_mask])
        mean_intensity = np.mean(rgb_region)

        feature = None

        if std_dev > 22 and edge_density > 0.15 and mean_intensity < 80:
            feature = "Damaged"
        elif std_dev > 18 and entropy > 7.5 and mean_intensity < 90:
            feature = "Molded"
        elif std_dev >= 15 and edge_density > 0.1:
            feature = "Rough Texture"
        elif std_dev >= 9 and edge_density > 0.11:
            feature = "Punctured"
        elif std_dev >= 12 and edge_density > 0.08:
            feature = "Blemished"
        elif std_dev >= 10 and edge_density < 0.08:
            feature = "Dents"
        elif std_dev > 5:
            feature = "Minor defects"
        elif std_dev > 2 and edge_density < 0.04:
            feature = "Smooth Texture"
        elif std_dev <= 2 and edge_density < 0.03:
            feature = "No defects"

        if feature:
            feature_scores[feature] += abs(importance)

    total_score = sum(feature_scores.values())
    if total_score > 0:
        for k in feature_scores:
            feature_scores[k] = round(100 * feature_scores[k] / total_score)

    return feature_scores

def display_feature_bar_graph(frame, feature_scores):
    for widget in frame.winfo_children():
        widget.destroy()
    features = list(feature_scores.keys())
    values = list(feature_scores.values())

    fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
    ax.barh(features, values, color="#e36a6a")
    for i, v in enumerate(values):
        ax.text(v + 3, i, str(v), color='black', va='center', fontweight='bold')

    ax.set_xlabel('Importance (%)')
    ax.set_title('Features', fontsize=14, color='black', pad=10, weight='bold')
    ax.invert_yaxis()
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    plt.subplots_adjust(left=0.30, right=0.90, top=0.80, bottom=0.25)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.config(bg="#f7f9f4", highlightthickness=0, bd=0)
    canvas_widget.pack(fill='both', expand=True, padx=0, pady=0)

def normalize_feature_scores(feature_scores):
    total = sum(feature_scores.values())
    if total == 0:
        return feature_scores
    normalized = {}
    for k, v in feature_scores.items():
        normalized[k] = round((v / total) * 100)
    diff = 100 - sum(normalized.values())
    if diff != 0:
        max_key = max(normalized, key=normalized.get)
        normalized[max_key] += diff
    return normalized


def explain_lime(image_path, use_bg_removed=False):
    explainer = lime_image.LimeImageExplainer()
    if use_bg_removed:
        rgba, bg_path = remove_background_with_yolo(image_path)
        if rgba is None:
            raise ValueError("No object detected for background removal.")
        rgb_img = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        rgb_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    rgb_resized = cv2.resize(rgb_img, (224, 224))

    explanation = explainer.explain_instance(
        image=rgb_resized,
        classifier_fn=model_fn,
        top_labels=1,
        hide_color=0,
        num_samples=350
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label=top_label,
        positive_only=True,
        hide_rest=False,
        num_features=5,
        min_weight=0.01
    )

    lime_img = mark_boundaries(rgb_resized, mask)

    os.makedirs("lime_outputs", exist_ok=True)
    base_name = os.path.basename(image_path)
    output_path = os.path.join("lime_outputs", f"lime_{base_name}")
    plt.imsave(output_path, lime_img)

    results = model(rgb_resized, verbose=False)
    boxes = results[0].boxes
    if boxes and boxes.shape[0] > 0:
        pred = results[0].names[int(boxes.cls[0])]
        conf = float(boxes.conf[0]) * 100
    else:
        pred, conf = "No object", 0

    return output_path, pred, conf

# ------------------------------
# Logging
# ------------------------------
def log_result(image_path, pred_class, confidence, user_confirmed,
               corrected_label="", lime_path="", major_contributor="",
               feature_scores=None):
    if user_confirmed:
        log_file = "logs/correct_predictions.csv"
        fields = ["image_path", "predicted_class", "confidence", "timestamp"]
        row = [image_path, pred_class, f"{confidence:.2f}", datetime.now().isoformat()]
    else:
        log_file = "logs/incorrect_predictions.csv"
        fields = [
            "image_path", "predicted_class", "confidence", "user_confirmed",
            "corrected_label", "lime_explanation_path", "timestamp",
            "major_contributor", "feature_scores"
        ]
        row = [
            image_path, pred_class, f"{confidence:.2f}", user_confirmed,
            corrected_label, lime_path, datetime.now().isoformat(),
            major_contributor,
            json.dumps(feature_scores) if feature_scores else ""
        ]

    os.makedirs("logs", exist_ok=True)
    is_new = not os.path.exists(log_file)
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(fields)
        writer.writerow(row)

def reset_globals():
    global img_path, pred, conf, last_lime_path, last_removed_bg_path
    global major_feature, feature_importance

    img_path = ""
    pred = ""
    conf = 0
    last_lime_path = ""
    last_removed_bg_path = ""
    major_feature = ""
    feature_importance = {}

# ------------------------------
# Tkinter Wizard GUI
# ------------------------------
class MangoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Carabao Mango Grading")
        self.geometry("800x480")
        self.configure()
        self.resizable(False, True)

        self.frames = {}
        for F in (StartFrame, PredictionFrame, CropFrame, BgRemovalFrame,
                  CorrectionFrame, LimeSelectionFrame, LimeResultFrame, CameraFrame):
            page_name = F.__name__
            frame = F(parent=self, controller=self)
            self.frames[page_name] = frame
            frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.show_frame("StartFrame")

    def show_frame(self, page_name):
        if page_name == "StartFrame":
            reset_globals()
            self.frames["PredictionFrame"].reset_display()
        frame = self.frames[page_name]

        if page_name == "PredictionFrame":
            frame.update_display()

        frame.tkraise()

# ------------------------------
# Individual Frames
# ------------------------------
class StartFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="white")
        self.controller = controller

        left_frame = tk.Frame(self, bg="white")
        left_frame.pack(side="left", fill="both", expand=False)

        try:
            bg_img = Image.open("image-removebg-preview (2).png")
            bg_img = bg_img.resize((400, 480))
            self.bg_photo = ImageTk.PhotoImage(bg_img)
            tk.Label(left_frame, image=self.bg_photo, bg="white").pack(fill="both", expand=True)
        except:
            tk.Label(left_frame, text="(Image missing)", bg="white").pack()

        right_frame = tk.Frame(self, bg="white")
        right_frame.pack(side="right", fill="both", expand=True, padx=40, pady=40)

        tk.Label(right_frame,
                 text="Welcome!",
                 font=("Arial Black", 25),
                 bg="white",
                 fg="black").pack(pady=10)

        tk.Label(right_frame,
                 text="This app aims to classify the grading of green carabao mango using YOLOv8. "
                      "If the image is misclassified, LIME will explain the features YOLOv8 relied on.\n\n"
                      "To do this, you can either upload or capture an image.",
                 font=("Arial", 13),
                 bg="white",
                 wraplength=300,
                 justify="left").pack(pady=10)

        tk.Button(
            self, text="Upload Image",
            command=self.upload_image,
            bg="#b9dab6", fg="black", font=("Arial", 14, "bold"), width=20, height=2
        ).place(x=480, y=280)

        tk.Button(
            self, text="Capture Image",
            command=lambda: controller.show_frame("CameraFrame"),
            bg="#b9dab6", fg="black", font=("Arial", 14, "bold"), width=20, height=2
        ).place(x=480, y=360)

    def upload_image(self):
        global img_path, pred, conf
        img_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.png;*.jpeg")]
        )
        if not img_path:
            return

        pred, conf, plotted_img = detect_mango(img_path)

        os.makedirs("predictions", exist_ok=True)
        out_path = os.path.join("predictions", "predicted.png")
        cv2.imwrite(out_path, plotted_img)

        self.controller.show_frame("PredictionFrame")

    def capture_image(self):
        global img_path, pred, conf
        img_path = capture_from_camera()
        if not img_path:
            return
        pred, conf, plotted_img = detect_mango(img_path)
        os.makedirs("predictions", exist_ok=True)
        out_path = os.path.join("predictions", "predicted.png")
        cv2.imwrite(out_path, plotted_img)
        self.controller.show_frame("PredictionFrame")

class CameraFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="white")
        self.controller = controller

        bg_image = Image.open("Sun-nature-ppt-backgrounds.jpg")
        bg_image = bg_image.resize((1150, 700))
        self.bg_tk = ImageTk.PhotoImage(bg_image)
        bg_label = tk.Label(self, image=self.bg_tk)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.preview_label = tk.Label(self, bg="#ffffff", text="Ready to capture using Pi Camera")
        self.preview_label.pack(pady=20)

        tk.Button(
            self, text="Capture Image",
            command=self.capture_image,
            bg="#b9dab6", fg="black", font=("Arial", 14, "bold"),
            width=15, height=2
        ).place(x=150, y=350)

        tk.Button(
            self, text="Back",
            command=lambda: controller.show_frame("StartFrame"),
            bg="#add8e6", fg="black", font=("Arial", 14, "bold"),
            width=15, height=2
        ).place(x=450, y=350)

    def capture_image(self):
        global img_path, pred, conf

        try:
            os.makedirs("camera_captures", exist_ok=True)
            img_path = os.path.join("camera_captures", "captured_image.jpg")

            messagebox.showinfo("Camera", "Opening Pi Camera...\nPreviewing for 2 seconds.")
            exit_code = os.system(f"rpicam-jpeg -o {img_path} -t 2000")

            if exit_code != 0 or not os.path.exists(img_path):
                messagebox.showerror("Camera Error", "Failed to capture image using rpicam-jpeg.")
                return

            pred, conf, plotted_img = detect_mango(img_path)
            os.makedirs("predictions", exist_ok=True)
            out_path = os.path.join("predictions", "predicted.png")
            cv2.imwrite(out_path, plotted_img)

            messagebox.showinfo("Capture Complete", "Image captured and processed.")
            self.controller.show_frame("PredictionFrame")

        except Exception as e:
            messagebox.showerror("Camera Error", f"An error occurred:\n{e}")

class PredictionFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        bg_image = Image.open("Sun-nature-ppt-backgrounds.jpg")
        bg_image = bg_image.resize((1150, 700))
        self.bg_tk = ImageTk.PhotoImage(bg_image)

        bg_label = tk.Label(self, image=self.bg_tk)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.image_panel = tk.Label(self, bg="#ffffff")
        self.image_panel.pack(pady=20)

        self.result_label = tk.Label(
            self, text="YOLOv8 Prediction: ---",
            font=("Arial", 24, "bold"), bg="#ffffff"
        )
        self.result_label.pack(pady=5)

        tk.Button(
            self, text="Correct",
            command=self.log_correct,
            bg="#b9dab6",
            fg="black",
            font=("Arial", 12, "bold"),
            width=15,
            height=2
        ).place(x=100, y=350)

        tk.Button(
            self, text="Incorrect",
            command=lambda: controller.show_frame("CropFrame"),
            bg="#f2b5a0",
            fg="black",
            font=("Arial", 12, "bold"),
            width=15,
            height=2
        ).place(x=300, y=350)

        tk.Button(
            self, text="Back",
            command=lambda: controller.show_frame("StartFrame"),
            bg="#add8e6",
            fg="black",
            font=("Arial", 12, "bold"),
            width=15,
            height=2
        ).place(x=500, y=350)

        self.bind("<<ShowFrame>>", self.on_show)

    def update_display(self):
        global pred, conf
        out_path = os.path.join("predictions", "predicted.png")
        if not os.path.exists(out_path):
            return

        show_image(out_path, self.image_panel, max_size=(300, 300))

        if pred and conf is not None:
            self.result_label.config(text=f"YOLOv8 Prediction: {pred} ({conf:.2f} %)")
        else:
            self.result_label.config(text="YOLOv8 Prediction: ---")

    def on_show(self, event=None):
        global pred, conf
        out_path = os.path.join("predictions", "predicted.png")
        if os.path.exists(out_path):
            show_image(out_path, self.image_panel)
            self.result_label.config(text=f"YOLOv8 Prediction: {pred} ({conf:.2f} %)")

    def log_correct(self):
        global img_path, pred, conf
        if not img_path:
            messagebox.showwarning("No image", "Please upload an image first.")
            return
        log_result(
            image_path=img_path, pred_class=pred, confidence=conf,
            user_confirmed=True
        )
        messagebox.showinfo("Logged", "Saved to correct_predictions.csv")
        self.controller.show_frame("StartFrame")

    def reset_display(self):
        self.image_panel.config(image="")
        self.image_panel.image = None
        self.result_label.config(text="YOLOv8 Prediction: ---")

class CropFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#f2d4d4")
        self.controller = controller

        bg_image = Image.open("Sun-nature-ppt-backgrounds.jpg")
        bg_image = bg_image.resize((1150, 700))
        self.bg_tk = ImageTk.PhotoImage(bg_image)
        bg_label = tk.Label(self, image=self.bg_tk)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.image_panel = tk.Label(self, bg="#ffffff", bd=0, highlightthickness=0)
        self.image_panel.place(x=50, y=70)

        self.crop_panel = tk.Label(self, bd=0, highlightthickness=0)
        self.crop_panel.place(x=380, y=70)

        tk.Label(self, text="Crop Image", bg="#f7f9f4", font=("Arial", 20, "bold")).pack(pady=20)

        tk.Button(
            self, text="Crop Image",
            command=self.crop_and_display_image,
            bg="#b9dab6",
            fg="black",
            font=("Arial", 12, "bold"),
            width=15,
            height=2
        ).place(x=200, y=350)

        tk.Button(
            self, text="Back",
            command=lambda: controller.show_frame("PredictionFrame"),
            bg="#add8e6",
            fg="black",
            font=("Arial", 12, "bold"),
            width=15,
            height=2
        ).place(x=400, y=350)

    def tkraise(self, *args, **kwargs):
        super().tkraise(*args, **kwargs)
        pred_path = os.path.join("predictions", "predicted.png")
        if os.path.exists(pred_path):
            show_image(pred_path, self.image_panel, max_size=(300, 300))

    def crop_and_display_image(self):
        if not img_path:
            messagebox.showwarning("No image", "Please upload an image first.")
            return
        cropped_img = crop_image_to_yolo_box(img_path)
        if cropped_img is None:
            messagebox.showwarning("No object", "No object to crop.")
            return

        pil_crop = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        pil_crop.thumbnail((300, 300))
        tk_crop = ImageTk.PhotoImage(pil_crop)
        self.crop_panel.config(image=tk_crop)
        self.crop_panel.image = tk_crop

        messagebox.showinfo("Success", "Cropped image ready. Proceeding to background removal...")
        self.controller.show_frame("BgRemovalFrame")

    def reset(self):
        self.image_panel.config(image="", text="")
        self.image_panel.image = None
        self.crop_panel.config(image="", text="")
        self.crop_panel.image = None

class BgRemovalFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        bg_image = Image.open("Sun-nature-ppt-backgrounds.jpg")
        bg_image = bg_image.resize((1150, 700))
        self.bg_tk = ImageTk.PhotoImage(bg_image)

        bg_label = tk.Label(self, image=self.bg_tk)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.pred_panel = tk.Label(self, bd=0, highlightthickness=0)
        self.pred_panel.place(x=50, y=70)

        self.bg_panel = tk.Label(self, bg="#fcfdfc", bd=0, highlightthickness=0)
        self.bg_panel.place(x=380, y=70)

        tk.Label(self, text="Remove Background Image", bg="#f7f9f4", font=("Arial", 20, "bold")).pack(pady=20)

        tk.Button(
            self, text="Remove Background",
            command=self.remove_background,
            bg="#b9dab6", fg="black", font=("Arial", 12, "bold"),
            width=20,
            height=2
        ).place(x=150, y=350)

        tk.Button(
            self, text="Back",
            command=lambda: controller.show_frame("CropFrame"),
            bg="#add8e6", fg="black", font=("Arial", 12, "bold"),
            width=15, height=2
        ).place(x=400, y=350)

    def tkraise(self, *args, **kwargs):
        super().tkraise(*args, **kwargs)
        pred_path = os.path.join("predictions", "predicted.png")
        if os.path.exists(pred_path):
            pil_img = Image.open(pred_path)
            pil_img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(pil_img)
            self.pred_panel.config(image=img_tk)
            self.pred_panel.image = img_tk

    def remove_background(self):
        global img_path, last_removed_bg_path

        if not img_path:
            messagebox.showwarning("No image", "Please upload an image first.")
            return

        rgba_image, saved_path = remove_background_with_yolo(img_path)

        pil_img = Image.fromarray(rgba_image).convert("RGBA")
        pil_img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(pil_img)
        self.bg_panel.config(image=img_tk)
        self.bg_panel.image = img_tk

        last_removed_bg_path = saved_path

        messagebox.showinfo("Success", "Background removed. Proceeding to correction...")
        self.controller.show_frame("CorrectionFrame")

    def reset(self):
        self.pred_panel.config(image="", text="")
        self.pred_panel.image = None
        self.bg_panel.config(image="", text="")
        self.bg_panel.image = None

class CorrectionFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        bg_image = Image.open("Sun-nature-ppt-backgrounds.jpg")
        bg_image = bg_image.resize((800, 480))
        self.bg_tk = ImageTk.PhotoImage(bg_image)
        bg_label = tk.Label(self, image=self.bg_tk)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        tk.Label(self, text="Select Corrected Class",
                 bg="#fcfcfc",
                 bd=0, highlightthickness=0,
                 font=("Arial", 20, "bold")
        ).place(x=240, y=150)

        self.correction = tk.StringVar(value="Extra Class")
        self.dropdown = ttk.Combobox(self, textvariable=self.correction, state="readonly", width=30)
        self.dropdown['values'] = ["Extra Class", "Class I", "Class II", "Class R"]
        self.dropdown.place(x=280, y=190)

        tk.Button(self, text="Next",
                  command=lambda: controller.show_frame("LimeSelectionFrame"),
                  bg="#b9dab6",
                  fg="black",
                  font=("Arial", 12, "bold"),
                  width=15,
                  height=2
        ).place(x=200, y=350)

        tk.Button(self, text="Back",
                  command=lambda: controller.show_frame("BgRemovalFrame"),
                  bg="#add8e6",
                  fg="black",
                  font=("Arial", 12, "bold"),
                  width=15,
                  height=2
        ).place(x=400, y=350)

    def reset(self):
        self.correction.set("Extra Class")

class LimeSelectionFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.progress_queue = Queue()
        self.elapsed_seconds = 0
        self.cycle_index = 0
        self.running = False

        bg_image = Image.open("Sun-nature-ppt-backgrounds.jpg").resize((800, 480))
        self.bg_tk = ImageTk.PhotoImage(bg_image)
        tk.Label(self, image=self.bg_tk).place(x=0, y=0, relwidth=1, relheight=1)

        tk.Label(
            self, text="Select image type for LIME",
            bg="#fcfcfc", font=("Arial", 20, "bold")
        ).place(x=200, y=150)

        self.lime_option = tk.StringVar(value="Original (uploaded image)")
        self.lime_menu = ttk.Combobox(
            self, textvariable=self.lime_option,
            state="readonly", width=40,
            values=[
                "Original (uploaded image)",
                "Cropped Image",
                "Removed Background Image",
                "Both Cropped and Removed Background Image"
            ]
        )
        self.lime_menu.place(x=250, y=190)

        tk.Button(
            self, text="Run LIME",
            command=self.run_lime_thread,
            bg="#b9dab6", fg="black",
            font=("Arial", 12, "bold"),
            width=15, height=2
        ).place(x=200, y=350)

        tk.Button(
            self, text="Back",
            command=lambda: controller.show_frame("CorrectionFrame"),
            bg="#add8e6", fg="black",
            font=("Arial", 12, "bold"),
            width=15, height=2
        ).place(x=400, y=350)

        self.loading_label = tk.Label(self, text="", bg="#fcfcfc", font=("Arial", 12, "bold"))
        self.loading_label.place(x=225, y=270)

        self.timer_label = tk.Label(self, text="", bg="#fcfcfc", font=("Arial", 12))
        self.timer_label.place(x=230, y=290)

        self.progress_bar = ttk.Progressbar(
            self, orient="horizontal", length=300, mode="determinate"
        )
        self.progress_bar.place(x=230, y=310)
        self.progress_bar.place_forget()

        self.message_cycle = [
            "Initializing LIME...",
            "Classify segments",
            "Assigning feature scores",
            "Determine major contributed feature",
            "Normalize into percentage",
            "Generating Bar Graph",
            "Highlighting influential areas from image",
            "Generating image heatmaps",
        ]

    def run_lime_thread(self):
        self.progress_bar.place(x=230, y=310)
        self.progress_bar["value"] = 0
        self.loading_label.config(text=self.message_cycle[0])
        self.timer_label.config(text="0:00")
        self.update_idletasks()

        self.elapsed_seconds = 0
        self.cycle_index = 0
        self.running = True

        threading.Thread(target=self.run_lime_process, daemon=True).start()
        self.after(1000, self.update_timer)
        self.after(100, self.update_progress_bar)

    def update_timer(self):
        if not self.running:
            return

        self.elapsed_seconds += 1
        minutes = self.elapsed_seconds // 60
        seconds = self.elapsed_seconds % 60
        self.timer_label.config(text=f"{minutes}:{seconds:02d}")

        if self.elapsed_seconds % 15 == 0:
            self.cycle_index = (self.cycle_index + 1) % len(self.message_cycle)
            self.loading_label.config(text=self.message_cycle[self.cycle_index])

        self.after(1000, self.update_timer)

    def update_progress_bar(self):
        try:
            progress = self.progress_queue.get_nowait()
            self.progress_bar["value"] = progress
            self.update_idletasks()
        except:
            pass

        if self.progress_bar["value"] < 100 and self.running:
            self.after(200, self.update_progress_bar)
        else:
            self.running = False
            self.loading_label.config(text="LIME Analysis Finalized and Completed!")
            self.after(800, self.finish_lime_ui)

    def finish_lime_ui(self):
        self.progress_bar.place_forget()
        self.loading_label.config(text="")
        self.timer_label.config(text="")
        self.controller.show_frame("LimeResultFrame")

    def run_lime_process(self):
        global last_lime_path, major_feature, feature_importance, img_path, last_removed_bg_path

        try:
            for i in range(10, 100, 10):
                time.sleep(0.3)
                self.progress_queue.put(i)

                if i == 20:
                    self.loading_label.config(text="Converting image to grayscale...")
                elif i == 40:
                    self.loading_label.config(text="Detecting edges...")
                elif i == 60:
                    self.loading_label.config(text="Compute texture and color metrics...")

            if not img_path:
                messagebox.showwarning("No image", "Please upload an image first.")
                self.progress_queue.put(100)
                return

            image_for_lime = img_path
            option = self.lime_option.get()
            if option == "Removed Background Image":
                if last_removed_bg_path:
                    image_for_lime = last_removed_bg_path
                else:
                    messagebox.showwarning("Missing", "Please run background removal first.")
                    self.progress_queue.put(100)
                    return
            elif option == "Cropped Image":
                cropped_img = crop_image_to_yolo_box(img_path)
                if cropped_img is None:
                    messagebox.showwarning("No object", "Could not detect object to crop.")
                    self.progress_queue.put(100)
                    return
                image_for_lime = save_temp_cropped_image(cropped_img)
            elif option == "Both Cropped and Removed Background Image":
                if not last_removed_bg_path:
                    messagebox.showwarning("Missing", "Please run background removal first.")
                    self.progress_queue.put(100)
                    return
                cropped_img = crop_image_to_yolo_box(last_removed_bg_path)
                if cropped_img is None:
                    messagebox.showwarning("No object", "Could not detect object to crop.")
                    self.progress_queue.put(100)
                    return
                image_for_lime = save_temp_cropped_image(cropped_img)

            explainer = lime_image.LimeImageExplainer()
            img = cv2.imread(image_for_lime, cv2.IMREAD_UNCHANGED)
            rgb = img[:, :, :3] if img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb, (250, 250))

            self.progress_queue.put(85)
            explanation = explainer.explain_instance(
                image=rgb_resized, classifier_fn=model_fn,
                top_labels=1, hide_color=0, num_samples=350
            )

            
            top_label = explanation.top_labels[0]
            lime_weights = dict(explanation.local_exp[top_label])  # {seg_id: weight}

            temp, mask = explanation.get_image_and_mask(
                top_label, positive_only=False,
                num_features=4, hide_rest=False
            )

            os.makedirs("lime_outputs", exist_ok=True)
            base_name = os.path.basename(image_for_lime).split('.')[0]
            lime_path = os.path.join("lime_outputs", f"{base_name}_lime.png")
            fig, ax = plt.subplots()
            ax.imshow(mark_boundaries(temp, mask))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(lime_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            last_lime_path = lime_path
            segments = explanation.segments

        
            feature_importance = classify_segments(mask, segments, rgb_resized, lime_weights=lime_weights)

            nonzero_features = {k: v for k, v in feature_importance.items() if v > 0}
            if nonzero_features:
                max_value = max(nonzero_features.values())
                tied_features = [k for k, v in nonzero_features.items() if v == max_value]
                if len(tied_features) == 1:
                    major_feature = tied_features[0]
                else:
                    major_feature = ", ".join(tied_features)
            else:
                major_feature = "N/A"

            feature_importance = normalize_feature_scores(feature_importance)
            self.progress_queue.put(100)

        except Exception as e:
            messagebox.showerror("Error", f"LIME failed: {e}")
            self.progress_queue.put(100)

    def reset(self):
        self.lime_option.set("Original (uploaded image)")

class LimeResultFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        bg_image = Image.open("Sun-nature-ppt-backgrounds.jpg")
        bg_image = bg_image.resize((1100, 750))
        self.bg_tk = ImageTk.PhotoImage(bg_image)
        bg_label = tk.Label(self, image=self.bg_tk)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.lime_panel = tk.Label(self, bg="#fcfdfc")
        self.lime_panel.place(x=50, y=60)

        self.graph_frame = tk.Frame(self, bg="#f7f9f4")
        self.graph_frame.place(x=350, y=60, width=390, height=300)

        self.details_label = tk.Label(self, text="", bg="#f7f9f4", font=("Arial", 12, "bold"))
        self.details_label.place(x=50, y=315)

        tk.Label(self, text="LIME Analysis", bg="#f7f9f4", font=("Arial", 20, "bold")).pack(pady=20)

        tk.Button(self, text="Confirm & Log",
                  command=self.confirm_and_log,
                  bg="#b9dab6",
                  fg="black",
                  font=("Arial", 12, "bold"),
                  width=15,
                  height=2
        ).place(x=200, y=350)

        tk.Button(self, text="Back",
                  command=lambda: controller.show_frame("LimeSelectionFrame"),
                  bg="#add8e6",
                  fg="black",
                  font=("Arial", 12, "bold"),
                  width=15,
                  height=2
        ).place(x=400, y=350)

    def tkraise(self, *args, **kwargs):
        super().tkraise(*args, **kwargs)
        self.update_display()

    def update_display(self):
        global last_lime_path, pred, conf, major_feature, feature_importance
        if last_lime_path and os.path.exists(last_lime_path):
            lime_img = Image.open(last_lime_path)
            lime_img.thumbnail((250, 250))
            lime_tk = ImageTk.PhotoImage(lime_img)
            self.lime_panel.config(image=lime_tk)
            self.lime_panel.image = lime_tk

        details = f"Major Contributor: {major_feature}\n"
        self.details_label.config(text=details, justify="left")

        if feature_importance:
            display_feature_bar_graph(self.graph_frame, feature_importance)

    def confirm_and_log(self):
        global img_path, pred, conf, last_lime_path, major_feature, feature_importance

        corrected_label = self.controller.frames["CorrectionFrame"].correction.get()

        log_result(
            image_path=img_path,
            pred_class=pred,
            confidence=conf,
            user_confirmed=False,
            corrected_label=corrected_label,
            lime_path=last_lime_path,
            major_contributor=major_feature,
            feature_scores=feature_importance
        )
        messagebox.showinfo("Logged", "Correction submitted and logged.")
        reset_globals()

        self.controller.frames["CropFrame"].reset()
        self.controller.frames["BgRemovalFrame"].reset()
        self.controller.frames["CorrectionFrame"].reset()
        self.controller.frames["LimeSelectionFrame"].reset()
        self.controller.frames["LimeResultFrame"].reset()
        self.controller.frames["PredictionFrame"].reset_display()

        self.controller.show_frame("StartFrame")

    def reset(self):
        self.lime_panel.config(image="", text="")
        self.lime_panel.image = None
        self.details_label.config(text="")
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    app = MangoApp()
    app.mainloop()