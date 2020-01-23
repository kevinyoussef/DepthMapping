#!/usr/bin/env python
#from libfreenect.wrappers.python import freenect
import cv2
from libfreenect.wrappers.python import frame_convert2

#import frame_convert2
import freenect


cv2.namedWindow("DEPTH")
print('Press ESC in window to stop')

def get_depth():
     return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])

while 1:
    img = get_depth()
    cv2.imshow('Depth', img)
    if cv2.waitKey(1) == 112:
        print("screenshot taken")
        print("Enter filename")
        title = input()
        cv2.imwrite(title + '.png', img)

    if cv2.waitKey(5) == 27:
        break


