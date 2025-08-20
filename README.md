## Computer vision for dissolution of AOT in a Water-Decane-AOT system using task-based incremental learning
Automated segmentation of AOT within a system, calculating size in pixel area with a benchmark of 100% AOT - the initial detected amount. Produces segmented video, plots and CSV data to visualise the dissolution of AOT over time.

## Versions: 
1. UNet segmentation:
- Pretrained ResNet18 on ImageNet weights
- Task-Incremental Online Learning
- Memory Replay Buffers

2. YOLO segmentation:
- Using Ultralytics pipeline (restrictive)

Future Ideas:
- EWC Regularisation
- Forgetting score calculation using Fischer Matrix (started)

## Trained Model Weights
Downloadable from: (https://drive.google.com/drive/folders/1exHoOWDJipvwZr2eqmvYhEoa_8QTodJB?usp=sharing)

- best_complete.pt - YOLO model for segmenting AOT
- vial_best.pt - YOLO model for detecting Vials
- trained_unet.pt - UNet model for segmenting AOT

- Test video downloadable from: https://theuniversityofliverpool-my.sharepoint.com/:f:/r/personal/rafiahn_liverpool_ac_uk/Documents/AOT%20Dissolution?csf=1&web=1&e=gmdKsK

## Usage
1. Clone the repository
2. Install Requirements
   ```pip install -r requirements.txt```
4. Download model weights from the above link and move them into the cloned repo
   
### Training (with new data)
- Record new video data and run frame_extraction.py to extract frames from video (can be used for multiple vials recorded simultaneously by adjusting parameters)
- Annotate data (using Roboflow) and download in YOLOV11 format
- Run create_masks.py to create masks for each image within the dataset
  - Adjust variables (tasks, colours etc.) within both scripts for any new classes added
- Run training_unet.py (To be added)

### Running the GUI
```python gui.py```
- Prompted load video or use camera: select specific camera index or .mp4, .avi, or .mov file.
- Prompted to save results: annotated video is stored with overlays, and prompted to save the CSV file.


The use of YOLO from Ultralytics was made under the AGPL-3.0 License.
