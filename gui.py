import tkinter as tk
from tkinter import Label, Button, Entry, simpledialog, filedialog, messagebox, ttk
import cv2 as cv
from PIL import Image, ImageTk
import numpy as np
import time
import threading
from pygrabber.dshow_graph import FilterGraph
import unet_modular
import matplotlib.pyplot as plt
import pandas as pd
import platform

# Global Variables
cap = cv.VideoCapture("noFeed.jpg")
selected_frame = None

area_text = None
scale = None
aot_sizes = []
starting_size = 0.0

times = []
start_time = None
last_measurement_time = None
measurement_interval = 0
interval_pixels = []
timer_job = None
interval_ms = 1000
experiment_running = False
video_writer = None 

seg_unet = unet_modular.SEG_MODEL
vial_yolo = unet_modular.VIAL_MODEL

# GUI to get real vial dimensions
def get_vial_dimensions_gui(root):
    try:
        width = simpledialog.askfloat("Vial Width", "Enter real vial WIDTH in mm:", parent=root)
        height = simpledialog.askfloat("Vial Height", "Enter real vial HEIGHT in mm:", parent=root)
        if width is None or height is None:
            raise ValueError("Cancelled or invalid input.", parent=root)
        return width, height
    except Exception as e:
        messagebox.showerror("Error", str(e), parent=root)
        exit()

# Choose Input 
def choose_input():
    global cap
    choice = messagebox.askquestion("Input Source", "Run from live camera?", icon="question")
    
    if choice == "yes":
        if platform.system() == "Darwin":  # macOS
            idx_str = simpledialog.askstring(
                "Select Camera",
                "Enter camera index (usually 0 for built-in webcam, 1+ for external):"
            )
            if idx_str is None:
                return

            try:
                idx = int(idx_str)
            except ValueError:
                return messagebox.showerror("Invalid selection", "Please enter a valid index.")

            cap.release()
            cap = cv.VideoCapture(idx, cv.CAP_AVFOUNDATION)
        else:  # Windows/Linux
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
            cap = cv.VideoCapture(idx)

        if not cap.isOpened():
            return messagebox.showerror("Camera Error", f"Could not open camera {idx}")

    else:
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video", "*.mp4 *.avi *.mov")]
        )
        if not path:
            return
        cap.release()
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            return messagebox.showerror("File Error", "Could not open selected video file.")

# Feed Update
def update_feed():
    global selected_frame
    
    ret, frame = cap.read()
    imgtk = None
    if ret:
        frame = cv.resize(frame, (128, 128))
        selected_frame = frame.copy()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_panel.imgtk = imgtk
        video_panel.configure(image=imgtk)

        if experiment_running:
            _, annotated_frame, _, _ = unet_modular.segment_frame(selected_frame)
            segmented_img_rgb = cv.cvtColor(annotated_frame, cv.COLOR_BGR2RGB)
            img = Image.fromarray(segmented_img_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            segmented_panel.imgtk = imgtk
            segmented_panel.configure(image=imgtk)

        # Write to video if writer exists
        if video_writer:
            video_writer.write(annotated_frame)
    
    video_panel.after(10, update_feed)


# Segmentation Function
def segment():
    global area_text, T, scale, interval_pixels, starting_size, start_time, last_measurement_time, aot_sizes, interval_ms
    if selected_frame is None:
        messagebox.showerror("Error", "No frame selected.")
        return
    
    area_text = "No AOT detected"
    if not scale and starting_size == 0.0:
        scale = unet_modular.get_scale(selected_frame, VIAL_REAL_WIDTH_MM, VIAL_REAL_HEIGHT_MM)

    masks, annotated_frame, area_pixels, stir_area = unet_modular.segment_frame(selected_frame)

    # Display the segmented image (moved into update feed method)
    # segmented_img_rgb = cv.cvtColor(annotated_frame, cv.COLOR_BGR2RGB)
    # img = Image.fromarray(segmented_img_rgb)
    # imgtk = ImageTk.PhotoImage(image=img)
    # segmented_panel.imgtk = imgtk
    # segmented_panel.configure(image=imgtk)

    current_time = time.time()

    # Initialise starting size
    if starting_size == 0.0 and area_pixels > 0:
        starting_size = area_pixels
        start_time = current_time
        last_measurement_time = start_time
        interval_pixels = [area_pixels]
        times.append(0)
        aot_sizes.append(area_pixels)
        print(f"Starting AOT size set: {starting_size} px²")
        T.delete("1.0", tk.END)
        T.insert(tk.END, f"Starting AOT size set: {starting_size} px²")

    if starting_size > 0:
        interval_pixels.append(area_pixels)
        if current_time - last_measurement_time >= measurement_interval:
            avg_area = int(np.mean(interval_pixels))
            elapsed_minutes = (current_time - start_time) / 60
            times.append(elapsed_minutes)
            aot_sizes.append(avg_area)
            last_measurement_time = current_time
            interval_pixels = []

            # Update GUI
            T.delete("1.0", tk.END)
            T.insert(tk.END, f"[{elapsed_minutes:.1f} min] AOT size: {avg_area} px²\nStirring Cylinder: {stir_area} px²")

    if (unet_modular.check_dissolved(aot_sizes, interval_ms)):
        finish = simpledialog.askfloat("AOT Dissolved", "No AOT has been detected for 20 minutes, we assume the AOT has dissolved\n Finish Experiment?")
        if finish:
            finish_experiment()


def start_timer():
    global timer_job, interval_ms, start_time, measurement_interval, experiment_running, video_writer
    experiment_running = True
    # Create VideoWriter
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    filepath = filedialog.asksaveasfilename(defaultextension=".mp4",
                                            filetypes=[("MP4 files", "*.mp4")],
                                            title=f"Enter Filename to Save Annotated Video")
    if filepath:
        height, width, _ = selected_frame.shape
        video_writer = cv.VideoWriter(filepath, fourcc, 20.0, (width, height))
    
    try:
        value = float(interval_entry.get())
    except ValueError:
        tk.messagebox.showerror("Invalid interval", "Enter a number.")
        return

    unit = unit_var.get()
    multiplier = {"seconds":1, "minutes":60, "hours":3600}[unit]
    measurement_interval = value * multiplier
    interval_ms = int(measurement_interval * 1000)

    # Disable start button, enable stop
    start_timer_btn.config(state="disabled")
    stop_timer_btn.config(state="normal")

    # Schedule first capture immediately, then every interval
    segment()
    timer_job = root.after(interval_ms, lambda: recurring_capture(interval_ms))
    stop_timer_btn.config(state="normal")
    plot_btn.config(state="normal")

def recurring_capture(interval_ms):
    global timer_job
    segment()
    timer_job = root.after(interval_ms, lambda: recurring_capture(interval_ms))

def stop_timer():
    global timer_job, experiment_running, video_writer
    experiment_running = False
    if timer_job:
        root.after_cancel(timer_job)
        timer_job = None
    start_timer_btn.config(state="normal")
    stop_timer_btn.config(state="disabled")

    # Release video writer if exists
    if video_writer:
        video_writer.release()
        video_writer = None
        messagebox.showinfo("Saved", "Annotated video saved successfully.")

def plot():
    global interval_ms, aot_sizes
    if not aot_sizes:
        messagebox.showerror("No Data", "No AOT size data to plot.")
        return

    # X-axis values: time in seconds
    interval_m = interval_ms / 60000 if interval_ms > 0 else 1
    time_points = [i * interval_m for i in range(len(aot_sizes))]
    aot_sizes_plot = aot_sizes[3:]
    time_points_plot = time_points[3:]

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(time_points_plot, aot_sizes_plot, marker='.', linestyle='-', color='b')
    plt.xlabel("Time (minutes)")
    plt.ylabel("AOT Size (px²)")
    plt.title("AOT Dissolution Over Time")

    # Plot percentage change
    percent = [(size / aot_sizes_plot[0]) * 100 for size in aot_sizes_plot]
    plt.subplot(1, 2, 2)
    plt.plot(time_points_plot, percent, marker='.', color='orange')
    plt.title("AOT Size Change (%) Over Time")
    plt.xlabel("Time (minutes)") 
    plt.ylabel("Change (%)")

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def reset_experiment():
    global aot_sizes, times, starting_size, start_time, experiment_running
    experiment_running = False
    aot_sizes.clear()
    times.clear()
    starting_size = 0.0
    start_time = None
    plot_btn.config(state="disabled")
    stop_timer_btn.config(state="disabled")

    video_panel.configure(image="")
    video_panel.imgtk = None
    segmented_panel.configure(image="")
    segmented_panel.imgtk = None

    T.delete("1.0", tk.END)
    T.insert(tk.END, "Experiment reset.\n")


def save_data():
    if not aot_sizes:
        messagebox.showerror("No Data", "No results to save.")
        return
    df = pd.DataFrame({"Time (min)": times, "AOT Size (px²)": aot_sizes})
    filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")],  title=f"Enter Filename to Save CSV")
    if filepath:
        df.to_csv(filepath, index=False)
        messagebox.showinfo("Saved", f"Results saved to {filepath}")

def finish_experiment():
    global experiment_running
    experiment_running = False
    if not aot_sizes:
        messagebox.showerror("No Data", "No experiment data to finish.")
        return
    
    stop_timer()
    plot()
    save_data()
    reset_experiment()

# GUI Setup
root = tk.Tk()
root.title("AOT Dissolution Tracking")

VIAL_REAL_WIDTH_MM, VIAL_REAL_HEIGHT_MM = get_vial_dimensions_gui(root)

source_button = ttk.Button(root, text="Choose Input Source", command=choose_input)
source_button.pack(side="top", fill="x", padx=10, pady=10)

video_panel = Label(root)
video_panel.pack(side="left", padx=10, pady=10)

segmented_panel = Label(root)
segmented_panel.pack(side="right", padx=10, pady=10)

segment_button = ttk.Button(root, text="Segment and Compute Size of AOT", command=segment)
segment_button.pack(side="top", fill="x", padx=10, pady=10)

reset_btn = ttk.Button(root, text="Reset Experiment", command=reset_experiment)
reset_btn.pack(side="bottom", pady=5)

save_btn = ttk.Button(root, text="Save Data", command=save_data)
save_btn.pack(side="bottom", pady=5)

finish_btn = ttk.Button(root, text="Finish Experiment", command=finish_experiment)
finish_btn.pack(side="bottom", pady=10)

T = tk.Text(root, height = 4, width = 45)
T.pack()
T.insert(tk.END, f"Info Here:")

# Timer setup widgets
tk.Label(root, text="Interval (AOT Size Calculated at):").pack(side="top", pady=(10,0))
interval_entry = tk.Entry(root)
interval_entry.insert(0, "30")    # default = 30
interval_entry.pack(side="top", padx=10)

unit_var = tk.StringVar(value="seconds")
unit_menu = ttk.OptionMenu(root, unit_var, "seconds", "seconds", "minutes", "hours")
unit_menu.pack(side="top", pady=5)

start_timer_btn = ttk.Button(root, text="Start Timer", command=lambda: start_timer(),state="normal")
start_timer_btn.pack(side="top", pady=5)

stop_timer_btn = ttk.Button(root, text="Stop Timer", command=lambda: stop_timer(), state="disabled")
stop_timer_btn.pack(side="top", pady=5)

plot_btn = ttk.Button(root, text="Plot Size of AOT", command=lambda: plot(), state="disabled")
plot_btn.pack(side="top", pady=5)

update_feed()
root.mainloop()
cap.release()
