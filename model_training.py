import os
import shutil
import random
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

tasks = [
    "datasets/10%-AOT-Dissolution-Tracking", 
    "datasets/30%-AOT-Dissolution-Tracking", 
    "datasets/50%-AOT-Dissolution-Tracking", 
    "datasets/70%-AOT-Dissolution-Tracking"
]

# Global replay buffer
replay_images = []
replay_labels = []
replay_val_images =[]
replay_val_labels = []

MAX_BUFFER_SIZE = 100 # test sample
PER_TASK_LIMIT = MAX_BUFFER_SIZE//len(tasks) # integer division
TASK_VAL_LIMIT = int(PER_TASK_LIMIT * 0.2)

def update_replay_buffer(new_task_dir):
    img_dir = os.path.join(new_task_dir, "train", "images")
    lbl_dir = os.path.join(new_task_dir, "train", "labels")
    img_val_dir = os.path.join(new_task_dir, "valid", "images")
    lbl_val_dir = os.path.join(new_task_dir, "valid", "labels")
    
    new_imgs = sorted(os.listdir(img_dir))
    new_lbls = sorted(os.listdir(lbl_dir))
    new_val_imgs = sorted(os.listdir(img_val_dir))
    new_val_lbls = sorted(os.listdir(lbl_val_dir))
    
    paired = list(zip(new_imgs, new_lbls))
    paired_val = list(zip(new_val_imgs, new_val_lbls))
    
    selected = random.sample(paired, min(PER_TASK_LIMIT, len(paired)))
    selected_val = random.sample(paired_val, min(TASK_VAL_LIMIT, len(paired_val)))
    
    # Add to replay buffer (test and validation images)
    for img, lbl in selected:
        replay_images.append(os.path.join(img_dir, img))
        replay_labels.append(os.path.join(lbl_dir, lbl))
        
    for img, lbl in selected_val:
        replay_val_images.append(os.path.join(img_val_dir, img))
        replay_val_labels.append(os.path.join(lbl_val_dir, lbl))

    # Trim buffer if over size
    if len(replay_images) > MAX_BUFFER_SIZE:
        idxs = random.sample(range(len(replay_images)), MAX_BUFFER_SIZE)
        replay_images[:] = [replay_images[i] for i in idxs]
        replay_labels[:] = [replay_labels[i] for i in idxs]
        

def build_replay_yaml(replay_dir, output_dir):
    img_dir = os.path.join(output_dir, "train", "images")
    lbl_dir = os.path.join(output_dir, "train", "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    img_val_dir = os.path.join(output_dir, "val", "images")
    lbl_val_dir = os.path.join(output_dir, "val", "labels")
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(lbl_val_dir, exist_ok=True)

    # Copy buffer data to temp folder
    for img_path, lbl_path in zip(replay_images, replay_labels):
        shutil.copy(img_path, img_dir)
        shutil.copy(lbl_path, lbl_dir)
    
    # Copy buffer data to temp folder
    for img_path, lbl_path in zip(replay_val_images, replay_val_labels):
        shutil.copy(img_path, img_val_dir)
        shutil.copy(lbl_path, lbl_val_dir)


    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""train: {os.path.abspath(img_dir)}
val: {os.path.abspath(img_val_dir)}
nc: 2
names: ['AOT', 'Stirring Cylinder']
""")
    return yaml_path


past_tasks = []

for i, task in enumerate(tasks):
    # Freeze backbone
    model.model.freeze = list(range(10))

    # Learn current task
    model.train(
        data=f"/mnt/scratch/users/sgrnade2/{task}/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name=f"task_{i}"
    )
    
    update_replay_buffer(f"/mnt/scratch/users/sgrnade2/{task}")

    # Fine-tune on replay buffer (if not empty)
    if len(replay_images) > 0:
        model = YOLO(f"runs/segment/task_{i}/weights/best.pt")
        # Build up replay dataset from past tasks
        replay_yaml = build_replay_yaml(
            replay_dir=replay_images,
            output_dir=f"replay_task_{i}"
        )
        
        model.train(
            data=replay_yaml,
            epochs=10,
            imgsz=640,
            batch=16,
            name=f"replay_task_{i}"
        )

    # Load best weights
    #model = YOLO(f"runs/segment/task_{i}/weights/best.pt")

    # Add this task to replay for future
    past_tasks.append(task)
