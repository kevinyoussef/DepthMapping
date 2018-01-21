import freenect
import cv2
from libfreenect.wrappers.python import frame_convert2


cv2.namedWindow('Video')
print('Press ESC in window to stop')

def get_video():
    print(frame_convert2.video_cv(freenect.sync_get_video()[0]))
    return frame_convert2.video_cv(freenect.sync_get_video()[0])

while 1:
    cv2.imshow('Video', get_video())
    if cv2.waitKey(10) == 27:
        break

