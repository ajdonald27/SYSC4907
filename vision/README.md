\# RATS – Vision Detection



\## Overview



This directory contains the camera-based UAV detection component of the Radar-Assisted Vision Tracking System (RATS). The objective is to compare YOLO11n and YOLO26n for UAV detection accuracy and real-time inference performance on NVIDIA hardware.



\## Directory Structure



\- `scripts/` – Python scripts for dataset conversion and annotation visualization.

\- `notebooks/` – Google Colab notebook used for model training and checkpoint recovery.

\- `results/` – Training configurations, performance metrics and trained model weights.

\- `data.yaml` – YOLO dataset configuration.

\- `requirements.txt` – Required Python dependencies.



\## Dataset



Both models were trained using the DUT Anti-UAV Detection and Tracking dataset.



Dataset source: https://github.com/wangdongdut/Anti-UAV-Detection-Tracking



The dataset contains one object class, UAV, and is divided into training, validation and testing subsets.



The dataset is not included in the repository due to its size. The converted dataset is available separately through Google Drive.



\### Dataset Preparation



If using the original dataset, place it inside the `vision/` directory using the following structure:



```

raw/

├── train/

│   ├── img/

│   └── xml/

├── val/

│   ├── img/

│   └── xml/

└── test/

&#x20;   ├── img/

&#x20;   └── xml/

```



From the `vision/` directory, install the dependencies:



```

pip install -r requirements.txt

```



Convert the original Pascal VOC XML annotations to YOLO format:



```

python scripts/convert\_dataset.py

```



The resulting dataset will have the following structure:



```

dataset/

├── images/

│   ├── train/

│   ├── val/

│   └── test/

└── labels/

&#x20;   ├── train/

&#x20;   ├── val/

&#x20;   └── test/

```



To visually verify the converted bounding boxes:



```

python scripts/visualize\_dataset.py

```



This script displays up to 25 randomly selected training images with their corresponding UAV bounding boxes.



If using the previously converted dataset ZIP, the conversion step can be skipped.



\## Model Training



YOLO11n and YOLO26n were initially trained using Google Colab with an NVIDIA Tesla T4 GPU.



| Parameter | Value |

|---|---|

| Epochs | 50 |

| Image size | 640 × 640 |

| Batch size | 8 |

| Random seed | 42 |

| Number of classes | 1 |

| Initial weights | Pretrained YOLO weights |



The original training notebook is located at `notebooks/RATS\_YOLO\_Training.ipynb`.



The notebook includes GPU verification, dataset extraction, training, checkpoint recovery and Google Drive integration. Colab-specific paths must be modified when running locally.



\## Training Results



The results for each model are stored in:



\- `results/yolo11n\_50epochs/`

\- `results/yolo26n\_50epochs/`



Each directory contains:



\- `args.yaml` – Training configuration.

\- `results.csv` – Training and validation metrics for each epoch.

\- `weights/best.pt` – Best-performing model checkpoint.

\- `weights/last.pt` – Most recent saved model checkpoint.



\### Validation Performance



| Metric | YOLO11n | YOLO26n |

|---|---:|---:|

| Precision | 95.36% | 95.71% |

| Recall | 80.38% | 85.69% |

| mAP@50 | 87.33% | 90.21% |

| mAP@50–95 | 53.16% | 56.68% |



YOLO26n achieved better validation performance across all four metrics after 50 epochs.



These results are based on the validation dataset. Independent test-set evaluation is still required.



\## Additional Training



The existing model weights can be used as the starting point for additional training on NVIDIA hardware.



The original 50-epoch results and weights should be preserved. Additional training experiments should be saved in separate directories.



A CUDA-compatible PyTorch installation is required for GPU training.



\## Future Work



\- Continue training on local NVIDIA hardware.

\- Evaluate both models using the held-out test dataset.

\- Compare inference latency and FPS on the Jetson Orin Nano.

\- Select the most suitable model for real-time UAV detection.

\- Integrate camera detections with radar measurements.

