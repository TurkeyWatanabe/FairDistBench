# Face4FairShifts Benchmark

This repository provides code for evaluating fairness-aware and robust facial attribute recognition using the Face4FairShifts dataset.


## 1. Environment Setup

Set up the Python environment using requirements.txt

`pip install -r requirements.txt`


## 2. Download the Dataset and Process

Download the dataset and place it in the appropriate folder. Set the excel_file_path and json_file_path in preprocessing.py, then run the script to generate face4fairshifts.json and place it in the anno folder. According to the options in the script arguments, the directory should be structured as follows (using our dataset as an example):

```
datasets/
└── face4fairshifts/
    ├── anno/
    │   └── Annotation.xlsx
    └── resized/
        └── *       ← contains cropped and resized face images
```

Make sure the directory structure matches exactly, as the code depends on this organization.

Set the `excel_file_path` and `json_file_path` variables in `preprocessing.py`, then run the script to generate `face4fairshifts.json` and place it in the anno folder.


## 3. Run the Code
Set the parameters and run python main.py, for example:
`python3 main.py --dataset=face4fairshifts --model=energy --label=age --task=oodd-a --domain=style --epoch=20`


### Supported Tasks:
+ fair: Fairness-aware learning
+ oodg: Domain generalization
+ oodd-s: OOD detection (sensory shift)
+ oodd-a: OOD detection (intra-domain semantic shift)
+ oodd-e: OOD detection (inter-domain semantic shift)
+ fairdg: Fairness-aware domain generalization

