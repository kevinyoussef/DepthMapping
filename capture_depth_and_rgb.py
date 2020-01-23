#!/usr/bin/env python

import cv2
from libfreenect.wrappers.python import frame_convert2
import freenect
import os
import csv

def get_depth():
     return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])

def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])



def main():
    #device = freenect.open_device(freenect.init(), 0)

    while 1:
        depthIMG = get_depth()
        rgbIMG = get_video()
        cv2.imshow('Depth', depthIMG)
        cv2.imshow('Video', rgbIMG)
        if cv2.waitKey(1) == 112:
            break
    #freenect.close_device(device)
    print("screenshot taken")
    print("Enter filename")
    title = input()
    path = os.getcwd()
    os.mkdir(path + '/' + title)
    os.chdir(path + '/' + title)
    cv2.imwrite(title + '_RBG.png', rgbIMG)
    cv2.imwrite(title + '_Depth.png', depthIMG)
    print(f'\n \n \n \n \n  {depthIMG.shape}  \n \n \n \n \n')
    with open(title + '_Depth.csv', 'w+') as csv_file:
        file_writer = csv.writer(csv_file, delimiter = ',')
        for i in range(len(depthIMG)):
            file_writer.writerow(depthIMG[i])
    print(f'\n \n \n \n \n  DONE 1 \n \n \n \n \n')
    
    with open(title + '_RGB.csv', 'w+') as csv_file:
        file_writer = csv.writer(csv_file, delimiter = ',')
        for i in range(len(rgbIMG)):
            file_writer.writerow(rgbIMG[i])
            
    print(f'\n \n \n \n \n  DONE 2 \n \n \n \n \n')

    os.chdir(path)




if __name__ == '__main__':
    main()

