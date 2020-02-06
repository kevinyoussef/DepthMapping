#!/usr/bin/env python
import numpy as np
import cv2
from libfreenect.wrappers.python import frame_convert2
import freenect
import os
import csv

def get_depth():
     return freenect.sync_get_depth()[0]
    

def get_video():
    return freenect.sync_get_video()[0]



def main():
    #device = freenect.open_device(freenect.init(), 0)

    while 1:
        depth = get_depth()
        RGB = get_video()
        depthIMG = frame_convert2.pretty_depth_cv(depth)
        rgbIMG = frame_convert2.video_cv(RGB)
        depthCSV = depth
        rgbCSV = RGB
        
        cv2.imshow('Depth', depthIMG)
        cv2.imshow('Video', rgbIMG)
        if cv2.waitKey(1) == 112:
            break
    print("depth")
    print(depth)
    print(np.max(get_depth()))
    print(np.max(depth))

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
    with open(title + '_Depth_11.csv', 'w+') as csv_file:
        file_writer = csv.writer(csv_file, delimiter = ',')
        for i in range(len(depthCSV)):
            file_writer.writerow(depthCSV[i])
    print(f'\n \n \n \n \n  DONE 1 \n \n \n \n \n')
    

    with open(title + '_Depth_8.csv', 'w+') as csv_file:
        file_writer = csv.writer(csv_file, delimiter = ',')
        for i in range(len(depthIMG)):
            file_writer.writerow(depthIMG[i])

    '''    
    with open(title + '_RGB.csv', 'w+') as csv_file:
        file_writer = csv.writer(csv_file, delimiter = ',')
        for i in range(len(rgbCSV)):
            file_writer.writerow(rgbCSV[i])
            
    print(f'\n \n \n \n \n  DONE 2 \n \n \n \n \n')
    '''
    os.chdir(path)
    print("depth")
    print(depth)
    print(np.max(get_depth()))
    print(np.max(depth))
    print("RGB")
    print(RGB)
    print(np.max(RGB))


if __name__ == '__main__':
    main()

