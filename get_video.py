import freenect
import cv2
from libfreenect.wrappers.python import frame_convert2


cv2.namedWindow('Video')
print('Press ESC in window to stop')

def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])

while 1:
    img = get_video()
    cv2.imshow('Video', img)
    if cv2.waitKey(1) == 112:
        print("Screenshot taken")
        title = input()
        cv2.imwrite(title + '.png', img)

    if cv2.waitKey(5) == 27:
        break
