# Computer vision for dissolution of AOT in a Water-Decane-AOT system using task-based incremental learning

2 Versions: 
YOLO segmentation:
Using Ultralytics pipeline 

UNet segmentation:
- Pretrained ResNet18 on ImageNet weights
- Task-Incremental Online Learning
- Memory Replay Buffers

Future Ideas:
- EWC Regularisation (Forgetting score calculation using Fischer Matrix)

Trained Model Weights
Downloadable from: [LINK*](https://drive.google.com/drive/folders/1exHoOWDJipvwZr2eqmvYhEoa_8QTodJB?usp=sharing)

best_complete.pt - YOLO model for segmenting AOT

vial_best.pt - YOLO model for detecting Vials

trained_unet.pt - UNet model for segmenting AOT



The use of YOLO from Ultralytics was made under the AGPL-3.0 License.
