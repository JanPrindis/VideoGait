# ⚙️ AlphaPose Setup
 
 [🔗 **Official GitHub Repository**](https://github.com/MVIG-SJTU/AlphaPose) | [📄 **Paper**](https://arxiv.org/abs/2211.03375)
 
 AlphaPose is an incredibly accurate, state-of-the-art multiperson pose estimator. However, due to its strict non-commercial licensing, it is **not directly included** in this repository. You have to fetch and build it yourself!
 
 ## 📦 1. Installation

- Requires Python (3.11)
- Linux users follow [official installation instructions](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md) for dependencies and setup, the code setup stays the same.
- Windows users need to follow these steps

## Download AlphaPose
- Download the AlphaPose code from [GitHub](https://github.com/MVIG-SJTU/AlphaPose)
- Extract the downloaded .zip file into `detectors/alphapose`

### Create a working environment with Python 3.11
- If using standalone, otherwise just use the project's environment.

### Pytorch - CUDA 12.6
 ```bash
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
 ```

### Dependencies

 ```bash
 pip install cython
 pip install PyYAML
 ```

### Fix for HalpeCOCOTOOLS:
(https://github.com/MVIG-SJTU/AlphaPose/issues/1195)

```bash
pip install git+https://github.com/Ambrosiussen/HalpeCOCOAPI.git#subdirectory=PythonAPI
 ```

Inside `setup.py` replace
```python
def get_install_requires():
    install_requires = [
        'six', 'terminaltables', 'scipy',
        'opencv-python', 'matplotlib', 'visdom',
        'tqdm', 'tensorboardx', 'easydict',
        'pyyaml', 'halpecocotools',
        'torch>=1.1.0', 'torchvision>=0.3.0',
        'munkres', 'timm==0.1.20', 'natsort'
    ]
```

with
```python
def get_install_requires():
    install_requires = [
        'six', 'terminaltables', 'scipy',
        'opencv-python', 'matplotlib', 'visdom',
        'tqdm', 'tensorboardx', 'easydict',
        'pyyaml',
        'torch>=1.1.0', 'torchvision>=0.3.0',
        'munkres', 'timm==0.1.20', 'natsort'
    ]
```

[If you want to use CUDA, set this flag to True](https://github.com/MVIG-SJTU/AlphaPose/blob/master/setup.py#L124)
> force_compile = True


### Build
```bash
python setup.py build develop
```

### Fix deprecated usage np.Float
```bash
pip install --upgrade cython-bbox
```

## 🧠 2. Model Weights

**Yolo Detector**
- [Download](https://drive.google.com/file/d/1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC/view)
- Move to `detectors/alphapose/detector/yolo/data` (you have to create the folder)

**[YOLOX Detector](https://github.com/Megvii-BaseDetection/YOLOX)**
- [Download](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth)
- Move to `detectors/alphapose/detector/yolox/data`(you have to create the folder)

**Tracker**
- [Download](https://drive.google.com/file/d/1myNKfr2cXqiHZVXaaG8ZAq_U2UpeOLfG/view)
- Remove the '(1)' at the end of the filename
- Move to `detectors/alphapose/detector/yolox/data`(you have to create the folder)

**Model Weights**
- [Downloads](https://drive.google.com/file/d/1zZotfE3WsBe1BxKimlK56wwJuK9E4EDs/view)
- Move to `trackers/weights/` (you have to create the folder)
 
## 💡 3. Notes

You can get different models on the [AlphaPose GitHub repository](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md).

### Do a prayer in a religion of your choice, and it _should_ work now...
