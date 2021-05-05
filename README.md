# The Pigeonator

An AI driven pigeon scarer using a Raspberry Pi.

## Background
We have a pigeon problem. It's not just that these birds are so fat they can barely fly (the incredulity of seeing one take off brings back memories of watching "Dumbo" as a child or my first trip on a 747). Nor is it the piles of pigeon cr\*p on the walls and garden furniture or the incessant cooing that we hear down our bedroom chimney at 6am in the morning. No, for me, the final straw was the pond. 

A couple of years ago we build a small pond in the garden. It certainly attracts wildlife, which was the aim. There are definitely more (pleasant) insects and birds in the garden now and they are joy to watch. Unfortunately, the previously resident pigeons have taken to performing their daily ablutions in there too. I have no idea why, or where they've been, but every pigeon bath leaves a revolting and most unsightly slick of oil on the surface of the water. I could stand it no longer and something had to be done.

## Pigeonator Mk I

Despite their unpleasantness, I'm not one for causing the pigeons particular harm; I just want to scare them off. My first attempt was the purchase of a *Pest XT Jet Spray* from Amazon, which is a battery operated motion detector that connects to the garden hose. This emits a 5s spray of water over a 60 degree arc whenever motion is detected in the general area. It certainly worked and did keep the pigeons away. No more oil slicks! However, a couple of issues prevented it from being the complete end to the story:

* The motion detector would sense movement from trees and plants blowing in the wind. The constant triggering in these circumstances was (probably) an irritation to the neigbours and a waste of water. On windy days I'd therefore have to nip out and turn the sensitivity down. Of course, I'd also then have to remember to turn it back up - on several occasions the reminder to do this was the appearence of another slick on the surface of the pond.
* More importantly, perhaps, the detector was indiscriminate and *also scared away the very birds we wanted to encourage*.

A more specific sensor trigger was required.

## The Rise, Fall and Rise of AI

I did quite a lot of work with neural networks back in the 1980's. In those days, they were limited in effectiveness because the computers that could train them were low powered (IBM PCs - no GPUs) and there was usually a dearth of data available. That, coupled with the fact that business uses were restricted (because the networks could never explain *why* they did something) meant that my interest gradually waned.

Scroll forward to 2019 when my eldest son started a PhD in Machine Learning at the University of Bath. Now, he told me, AI and neural networks are back. It's just that today they are called *Deep Learning*. Also, now there is a ton of data and processing power available everywhere so they might actually be useful for something (you can see where I'm going with this, I think). They still can't explain the reasoning behind their decisionmaking but we'll not let that deter us; after all, if we squirt the neighbours cat rather than a pigeon, we won't lose that much sleep over it.

## Pigeonator Mk II

So the new plan is to create a pigeon scaring device that is sensitive only to these fat birds and not to other cute wildlife and the vagaries of the weather. Perhaps we can use AI image classification to do this? With the advent of TensorFlow (and other ML toolkits) it seems this should be possible and, what's more, this stuff can even run on a mobile, low powered, computer like a Raspberry Pi. Now we're talking! 

## Pi Setup

For use with the HQ camera, you need to bump the GPU memory in /boot/config.txt to 176Mb (https://www.raspberrypi.org/forums/viewtopic.php?t=278381)

* Install IFTTT webhook package:
  ```bash
  pip3 install git+https://github.com/DrGFreeman/IFTTT-Webhook.git
  ```
* Fetch Pigeonator
  mkdir ~/Projects
  cd ~/Projects
  git clone https://github.com/bowerhaus/Pigeonator.git
  ```
