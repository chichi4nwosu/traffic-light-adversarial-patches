##Improving Experiments on YOLOV8M trained of BSTLD Dataset

#1. Train Yolov8m model:
-------------------------------


Run the command on on terminal:


```bash
yolo detect train model=yolov8m.pt data=configs/bstld_yolo.yaml imgsz=1280 epochs=150 batch=8 patience=25 project=runs/rebuild name=bstld1280_v8m_long
```

This created: 

runs/rebuild/bstld1280_v8m_long/
  weights/
    best.pt      
    last.pt
  

runs/rebuild/bstld1280_v1/weights/best.pt. : this weight files will be used as the baseline for evaluating attacks and defenses



#2. Evaluating Clean Classification Accuracy of Yolov8m model:

The script named bstld-evaluate-clean-acc.py has the code to evaluate Clean Classification Accuracy

Before running the script I created a config file named bstld_config_v8m_long.yaml

Then Run the command:

```bash
python bstld-evaluate-clean-acc.py \
  configs/bstld_config_v8m_long.yaml \
  runs/rebuild/bstld1280_v8m_long/weights/best.pt
```

The output prints:
=== Clean TL Classification Accuracy (IoU-matched) ===
Total GT red TLs: 182
Detected (matched): 116
Correct color: 113
Misclassified: 3
Vanished: 66

-- Detection axis --
Detection recall: 0.6373626373626373
Vanishing rate: 0.3626373626373626

-- Classification axis (conditioned on correct detection) --
Conditional classification accuracy: 0.9741379310344828
Conditional misclassification rate: 0.02586206896551724

Saving clean evaluation bar chart to outputs-clean-eval-v8m/clean_eval_bar.png



