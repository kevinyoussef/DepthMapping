#!/usr/bin/env python
#from libfreenect.wrappers.python import freenect
import cv2
from libfreenect.wrappers.python import frame_convert2

#import frame_convert2
import freenect


cv2.namedWindow("DEPTH")
print('Press ESC in window to stop')

def get_depth():
     print(frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0]))
     return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])

while 1:
    cv2.imshow('Depth', get_depth())
    if cv2.waitKey(10) == 27:
        break

