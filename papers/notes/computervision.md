# Computer Vision Basics

## Object Detection

### What is it?

Object detection identifies both what an object is and where it is located in an image.

### Output

* Class label
* Confidence score
* Bounding box

### Why it Matters

Autonomous vehicles use object detection to identify:

* Traffic lights
* Stop signs
* Pedestrians
* Vehicles

### Resources

* YOLO (You Only Look Once)
* Faster R-CNN

---

## Bounding Boxes

### What is it?

A bounding box is a rectangle drawn around an object.

### Components

* x-coordinate
* y-coordinate
* width
* height

### Why it Matters

Object detectors use bounding boxes to show where an object is located.

---

## YOLO

### What is it?

YOLO (You Only Look Once) is a real-time object detection model.

### Advantages

* Fast
* Accurate
* Widely used

### Applications

* Autonomous vehicles
* Drones
* Traffic monitoring

### Why it Matters

The Fool the Stoplight paper attacks traffic light detection systems similar to YOLO-based detectors.

---

## CNN (Convolutional Neural Network)

### What is it?

A CNN is a neural network designed for image processing.

### What it Learns

* Edges
* Shapes
* Patterns
* Objects

### Why it Matters

Most computer vision models are built on CNN backbones.

---

## Training

### What is it?

Training is the process of teaching a model using labeled data.

### Process

Dataset → Prediction → Error → Update Weights → Repeat

### Key Terms

* Epoch
* Batch
* Learning Rate
* Loss Function

---

## Inference

### What is it?

Inference is when a trained model makes predictions on new data.

### Example

Image → Model → Traffic Light Detected

### Difference from Training

No learning occurs during inference.

---

## Datasets

### What is it?

Collections of labeled images used to train and evaluate models.

### Common Datasets

* ImageNet
* COCO
* BDD100K

### Why it Matters

The quality of a dataset strongly affects model performance.

---

## Adversarial Examples

### What is it?

Inputs intentionally modified to fool a machine learning model.

### Goal

Cause incorrect predictions while appearing normal to humans.

### Why it Matters

Shows that deep learning systems can be vulnerable to attack.

---

## Adversarial Patches

### What is it?

A visible pattern designed to fool a model when placed in an image or real-world environment.

### Characteristics

* Visible
* Physical
* Transferable

### Examples

* Traffic sign attacks
* Traffic light attacks
* Autonomous vehicle perception attacks

### Why it Matters

Fool the Stoplight uses adversarial patches to manipulate traffic light detectors.
