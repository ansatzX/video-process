# Video-Process for V-log 

## installation
```
pip install numpy matplotlib opencv-python mediapipe
```


## project-explain


mask face in each frame in your video,

Now, I write down some function for processing one picture

workflow can be 

1. use ffmpeg to generate png of each frame

2. proces each pic by python code

3. process pics to video without sound

4. ffmpeg merge video and sound

5. v-log is ok


## example

```
from videoprocess.process import process_a_pic

process_a_pic('./img.png','zxm')

```

We do not provide picture with human face, one can test with one's own picture.