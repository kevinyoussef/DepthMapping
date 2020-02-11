"""
ECE196 Depth Mapping Project
Author: Will Chen
Prerequisite: You need to install OpenCV before running this code
The code here is an example of what you can write to print out 'Hello World!'
Now modify this code to process a local image and do the following:
1. Read geisel.jpg
2. Convert color to gray scale
3. Resize to half of its original dimensions
4. Draw a box at the center the image with size 100x100
5. Save image with the name, "geisel-bw-rectangle.jpg" to the local directory
All the above steps should be in one function called process_image()
"""

# TODO: Import OpenCV
import cv2

# TODO: Edit this function
def process_image():
    img = cv2.imread('geisel.jpg',0)
    height = int(img.shape[0]/2)
    width = int(img.shape[1]/2)
    dim = (width, height)
    resized = cv2.resize(img, dim)
    start_point = (int(resized.shape[1]/2)-50, int(resized.shape[0]/2)-50)
    end_point = (int(resized.shape[1]/2)+50, int(resized.shape[0]/2)+50)
    color = (255,255,255)
    thickness = 5
    resized = cv2.rectangle(resized, start_point, end_point, color, thickness)
    cv2.imwrite("resized.jpg", resized)

# Just prints 'Hello World! to screen.
def hello_world():
    print('Hello World!')
    return

# TODO: Call process_image function.
def main():
    hello_world()
    process_image()
    return


if(__name__ == '__main__'):
    main()
