# OpenCV-Bloody-Gauze-Detection
A small image processing application that was planned to crop out the bloody gauze out of the picture.

# Drastically improved the blood detection
Blood detection is providing satisfactory results now and it reliably create a rectangle around the blood.
But it is still incomplete as it requires to crop out just a tiny bit out from the gauze, and the provided pictures are not uniform in angles.

# The models haven't been trained with the cropped parts
Instead they were trained by the hu moments and the histogram values of the images in grayscale color space.

# The tensorflow lite model is there just because
I spent some time training the model so I figured I'd also keep it
