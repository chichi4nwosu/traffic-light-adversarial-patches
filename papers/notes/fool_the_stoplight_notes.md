# Fool the Stoplight

## Problem

What are they trying to break?

## Model

What neural network are they attacking?

## Dataset

What images are used?

## Patch

How is the adversarial patch generated?

## Metrics

How do they measure success?

## Limitations

What weaknesses does the paper have?

## Future Work

What could be improved?

BSTLD

Bosch Small Traffic Lights Dataset.

Contains:

Red Lights
Yellow Lights
Green Lights
Off Lights

The model learns from these images.

Expectation Over Transformation

Instead of training a patch for:

One Image

Train for:

Brightness Changes
Rotation
Distance
View Angle

So the patch works in the real world.

The code literally changes:

Brightness
Rotation
Padding
Resize

during training.