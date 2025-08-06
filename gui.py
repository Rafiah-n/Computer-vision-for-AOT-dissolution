import tkinter as tk
from tkinter import Label, Button, Entry, simpledialog, filedialog, messagebox, ttk
import cv2 as cv
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
from pygrabber.dshow_graph import FilterGraph
import seg_vis

cap = cv.VideoCapture("noFeed.jpg")
selected_frame = None

# Choose Input 
def choose_input():
    global cap
    choice = messagebox.askquestion("Input Source", "Run from live camera?", icon="question")
    if choice == "yes":
        graph = FilterGraph()
        cams = graph.get_input_devices()
        if not cams:
            return messagebox.showerror("No Cameras", "No webcams found.")

        cam_name = simpledialog.askstring(
            "Select Camera",
            "Available cameras:\n" + "\n".join(f"{i}: {n}" for i, n in enumerate(cams)) +
            "\n\nEnter the index of your choice:"
        )
        if cam_name is None:
            return

        try:
            idx = int(cam_name)
            if idx < 0 or idx >= len(cams):
                raise ValueError
        except ValueError:
            return messagebox.showerror("Invalid selection", "Please enter a valid index.")

        cap.release()
        cap = cv.VideoCapture(idx, cv.CAP_DSHOW)

    else:
        path = filedialog.askopenfilename(title="Select video file", filetypes=[("Video", "*.mp4 *.avi *.mov")])
        if not path:
            return
        cap.release()
        cap = cv.VideoCapture(path)

# Feed Update
def update_feed():
    global selected_frame
    ret, frame = cap.read()
    if ret:
        selected_frame = frame.copy()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_panel.imgtk = imgtk
        video_panel.configure(image=imgtk)
    video_panel.after(10, update_feed)

# Segmentation Function
def segment():
    if selected_frame is None:
        messagebox.showerror("Error", "No frame selected.")
        return
    scale = seg_vis.get_scale(selected_frame, get_vial_dimensions_gui[0], get_vial_dimensions_gui[0])
    processed = seg_vis.segment_frame(selected_frame, scale)

    # Display the segmented image
    segmented_img_rgb = cv.cvtColor(processed, cv.COLOR_BGR2RGB)
    img = Image.fromarray(segmented_img_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    segmented_panel.imgtk = imgtk
    segmented_panel.configure(image=imgtk)

# GUI to get real vial dimensions
def get_vial_dimensions_gui():
    root = tk.Tk()
    root.withdraw()
    try:
        width = simpledialog.askfloat("Vial Width", "Enter real vial WIDTH in mm:")
        height = simpledialog.askfloat("Vial Height", "Enter real vial HEIGHT in mm:")
        if width is None or height is None:
            raise ValueError("Cancelled or invalid input.")
        return width, height
    except Exception as e:
        messagebox.showerror("Error", str(e))
        exit()


# GUI Setup
root = tk.Tk()
root.title("AOT Dissolution Tracking")

VIAL_REAL_WIDTH_MM, VIAL_REAL_HEIGHT_MM = get_vial_dimensions_gui()

source_button = ttk.Button(root, text="Choose Input Source", command=choose_input)
source_button.pack(side="top", fill="x", padx=10, pady=10)

video_panel = Label(root)
video_panel.pack(side="left", padx=10, pady=10)

segmented_panel = Label(root)
segmented_panel.pack(side="right", padx=10, pady=10)

segment_button = ttk.Button(root, text="Segment and Compute Size of AOT", command=segment)
segment_button.pack(side="bottom", fill="both", padx=10, pady=10)

update_feed()
root.mainloop()
cap.release()
