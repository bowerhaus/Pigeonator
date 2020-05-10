# The Pigeonator

An AI driven pigeon scarer .

## Background
We have a pigeon problem. It's not just that these birds are so fat they can barely fly. It's not the pigeon cr\*p on the walls and garden furniture. Nor is it the incessant cooing that we hear down our bedroom chimney. No, final straw was the pond. 

A couple of years ago we build a small pond in the garden. It certainly attracts wildlife, which was the aim. There are more (pleasant) insects and birds in the garden now and they are joy to watch. Unfortunately, the previously resident pigeons have also taken to performing their daily ablutions in there. I have no idea why, or where they've been, but every pidgeon bath leaves a disgusting and most unsightly slick of oil on the surface of the water. I could stand it no longer and something had to be done.

## Pigeonator Mk I

I'm not one for causing the pigeons particular harm; I just want to scare them off. My first attempt was the purchase of a Pest XT Jet Spray from Amazon, which is a battery operated motion detector that connects to the garden hose. This emits a 5s spray of water over a 60 degree arc whenever motion is detected in the general area. This certainly worked, and did keep the pigeons away; no more oil slicks! However, a couple of issues prevented it from being the end to the story:

* The motion detector will sense movement from trees and plants blowing in the wind. The constant triggering in these circumstances is (probably) an irritation to the neigbours and a waste of water. On windy days I therefore have to nip out and turn the sensitivity down. Of course, I also then have to remember to turn it back up - on several occasions the reminder to do this has been another slick on the pond surface.
* More importantly, perhaps, the detector is indiscriminate and *also scares away the very birds we want to encourage*.

A more specific sensor trigger was required.

## The Rise, Fall and Rise of AI

I did quite a lot of work with neural networks back in the 1980's. In those days, they were limited in effectiveness because the computers that could train them were low powered (IBM PCs - no GPUs) and there was usually a dearth of data available. That, coupled with the fact that business uses were limited because the network could never explain *why* they did someing, meant that my interest gradually waned.

Scroll forward to 2019 when my eldest son started a PhD in Machine Learning at the University of Bath. Now, it seems, AI and neural networks are back. It's just that today they are called *Deep Learning*. Also, there is a ton of data and processing power so they might actually be useful for something (you can see where I'm going with this, I'm sure). They still can explain their choices but we'll ignore that for now.

## Pigeonator Mk II

So the plan is to create a pigeon scaring device that is sensitive only to these fat birds and not to other cute wildlife and the vagaries of the weather. Perhaps we can use AI image classification to do this? With the advent of TensorFlow (and other ML toolkits) it seems this should be possible and, what's more, this stuff can even run on a mobile, low powered, computer like a Raspberry Pi. Now we're taking! 
