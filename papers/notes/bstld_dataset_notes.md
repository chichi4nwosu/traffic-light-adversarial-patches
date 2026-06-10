# BSTLD Dataset Notes

## Dataset Name

BSTLD (Bosch Small Traffic Lights Dataset)

---

## What is BSTLD?

The Bosch Small Traffic Lights Dataset (BSTLD) is a computer vision dataset designed for traffic light detection and classification in autonomous driving systems. The dataset was created to support research in perception systems for self-driving vehicles, particularly because traffic lights are often small, distant objects that are difficult to detect accurately.

The dataset contains images captured from a vehicle-mounted camera in real driving environments. Each traffic light is annotated with a bounding box and a traffic light state label. BSTLD has become one of the most commonly used datasets for traffic light detection research and is frequently used to evaluate autonomous vehicle perception systems.

---

## Dataset Size

Total Dataset:

* 13,427 images
* Approximately 24,000 annotated traffic lights
* Resolution: 1280 × 720 pixels

Training Set:

* 5,093 images
* 10,756 annotated traffic lights

Test Set:

* 8,334 images
* 13,486 annotated traffic lights

Traffic lights are often very small within the image, with a median width of approximately 8.5 pixels. This makes the dataset particularly challenging for object detection models.

---

## Traffic Light Classes

The project simplifies traffic light states into four classes:

* Red
* Yellow
* Green
* Off

These classes are used in Sumaiya's project and in the attack/defense experiments.

---

## Why is BSTLD Challenging?

Several factors make BSTLD difficult:

1. Traffic lights are extremely small.
2. Traffic lights may be partially occluded.
3. Lighting conditions vary significantly.
4. Traffic lights appear at different distances.
5. Images come from real driving environments rather than controlled laboratory settings.

Because of these challenges, BSTLD is considered a realistic benchmark for autonomous vehicle perception research.

---

## Why is BSTLD Good for This Research?

BSTLD is a good dataset for adversarial attack research because:

* It focuses specifically on traffic lights.
* Traffic lights are safety-critical objects.
* Small objects are generally harder to detect.
* Misclassification of traffic lights can directly impact autonomous vehicle behavior.
* The dataset contains realistic driving scenes.

In this project, BSTLD allows us to evaluate whether adversarial patches can cause YOLOv8 to misclassify traffic light states, such as changing a red light into a green light prediction.

---

## Connection to My Project

Dataset:
BSTLD

Model:
YOLOv8

Attack:
Adversarial Patch + EOT

Goal:
Cause traffic light misclassification while maintaining realistic physical-world conditions.

Research Question:
Can adversarial patches reliably fool traffic light recognition systems used in autonomous vehicles?

---

## Key Takeaways

* BSTLD is one of the most important traffic light datasets in autonomous driving research.
* It contains over 13,000 images and 24,000 annotated traffic lights.
* Traffic lights are very small and difficult to detect.
* The dataset is ideal for evaluating adversarial attacks on traffic light recognition systems.
* This dataset forms the foundation of the Fool the Stoplight and Sumaiya project experiments.
