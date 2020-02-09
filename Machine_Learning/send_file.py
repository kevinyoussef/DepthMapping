import subprocess, time, os

EC2_IP = 'ec2-ubuntu@ec2-54-188-163-113.us-west-2.compute.amazonaws.com'
KEY_PAIR_PATH = 'C:\\Users\\kevin\\Desktop\\Password Key Files\\GPYPainKey.pem'

IMG_DEST_DIR = '~\\DepthMapping\\Machine_Learning'
file_path = 'C:\\Users\\kevin\\Desktop\\ECE 196\\DepthMapping\\Machine_Learning\\test.py'

def send_file():
	print("sending file to server...")
	
	send_command = ["scp", "-i", KEY_PAIR_PATH, file_path, EC2_IP + ":" + IMG_DEST_DIR]
	subprocess.call(send_command)

	print("file successfully send!")


def fetch_result():
        print("fetching file to the sever...")

        ## write code below the print statement ##

if __name__ == '__main__':
	send_file()	
	fetch_result()	
