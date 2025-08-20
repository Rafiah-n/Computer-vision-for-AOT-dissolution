# Computer vision for dissolution of AOT in a Water-Decane-AOT system using task-based incremental learning
## Project Description:
Automated segmentation of AOT within a system, calculating size in pixel area with a benchmark of 100% AOT - the first detected amount. Produces segmented video, plots and CSV data to visualise the dissolution of AOT over time.

## 2 Versions: 
YOLO segmentation:
- Using Ultralytics pipeline (restrictive)

UNet segmentation:
- Pretrained ResNet18 on ImageNet weights
- Task-Incremental Online Learning
- Memory Replay Buffers

Future Ideas:
- EWC Regularisation (Forgetting score calculation using Fischer Matrix)

## Trained Model Weights
Downloadable from: (https://drive.google.com/drive/folders/1exHoOWDJipvwZr2eqmvYhEoa_8QTodJB?usp=sharing)

- best_complete.pt - YOLO model for segmenting AOT
- vial_best.pt - YOLO model for detecting Vials
- trained_unet.pt - UNet model for segmenting AOT

## Usage
1. Clone the repository
2. Install Requirements
   ```pip install -r requirements.txt```
4. Download model weights from the above link and move them into the
   
### Training (with new data)
- Annotate data (using Roboflow) and download in YOLOV11 format
- Run create_masks.py to create masks for each image within the dataset
  - Adjust variables (tasks, colours etc.) within both scripts for any new classes added
- Run training_unet.py (To be added)

### Running the GUI
```python gui_unet.py```
- Prompted load video or use camera: select specific camera index or .mp4, .avi, or .mov file.
- Prompted to save results: annotated video is stored with overlays, and prompted to save the CSV file.


The use of YOLO from Ultralytics was made under the AGPL-3.0 License.
