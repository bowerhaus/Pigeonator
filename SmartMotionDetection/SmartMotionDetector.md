# Smart AI-Driven Motion Detection for the Raspberry Pi

This is a part of the [Pigeonator](../README.md) project. The aim is to gather images of pigeons and other birds for use as training data for a TensorFlow model. The code is general purpose enough so that it can probably used, with some modification, for gathering other images or as the basis for an RPi-powered smart security scanner.

## Why not standard motion detection?

Before engaging on this project, I had tried a more traditional motion detection camera approach. Initially, I made use of a [Neos Smartcam](https://www.amazon.co.uk/Neos-SmartCam-Vision-Camera-Warranty/dp/B07JY7K3SZ) in a [waterproof housing](https://www.amazon.co.uk/gp/product/B07R2KD9BJ/ref=ppx_yo_dt_b_asin_title_o09_s00?ie=UTF8&psc=1) mounted to the garden greenhouse. This was immediately seen as inappropriate because the lens is too much of a wideangle and the image resolution was poor. After this, I tried a couple of RPi solutions, such as [MotionEyeOS](https://github.com/ccrisan/motioneyeos/wiki). This was extremely easy to set up and, with the [RPi 2.1 Camera](https://www.amazon.co.uk/gp/product/B01ER2SKFS/ref=ppx_yo_dt_b_asin_title_o02_s01?ie=UTF8&psc=1), yielded images of sufficient resolution but the tendency to trigger on false positive motion made it impractical. It was all to easy to end up with hundreds of photos of non-pigeons triggered by leaves blowing in the wind or shadows moving.

## Using TensorFlow Lite and the MobileNet model

The program here has its genesis in the [TensorFlow Object Detection](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/README.md) example for Raspberry Pi.
