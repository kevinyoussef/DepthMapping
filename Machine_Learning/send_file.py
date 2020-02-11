# currently, get the file 'hello_world.py' from current dir send it to AWS to dir 'source'
# Then, get the file from AWS 'result' dir and send it back locally to 'transferred_files' dir

import subprocess, time, os

EC2_IP = 'ec2-user@ec2-18-208-227-191.compute-1.amazonaws.com' # change to correct IP
KEY_PAIR_PATH = '../test_key_pair.pem' # change to PI's key-pair location

# to send data
AWS_DIR = '~/DepthMapping/Machine_Learning/'
FILE_SEND_PATH = './test.py' # need to be change to whatever file

# to fetch data
FILE_FETCH_PATH = 'result/test.py'      # change to correct file
FILE_DEST_DIR = './transferred_files/'

def send_file():
    print("sending file to server")

    send_command = ["scp", "-i", KEY_PAIR_PATH, FILE_SEND_PATH, EC2_IP + ":" + AWS_DIR + "source"]
    subprocess.call(send_command)

    print("file sent!")


def fetch_file():
    # create the folder if doens't exist
    if not os.path.exists('transferred_files'):
        os.makedirs('transferred_files')

    print("\ngetting the file from server")

    fetch_command = ["scp", "-i", KEY_PAIR_PATH, EC2_IP + ":" + AWS_DIR + FILE_FETCH_PATH, FILE_DEST_DIR]

    subprocess.call(fetch_command)


    print("file was received!")

def main():
    # creates folders in not exist already
    if not os.path.exists('source'):
        os.makedirs('source')
    if not os.path.exists('result'):
        os.makedirs('result')

    send_file()

    fetch_file()

if __name__ == "__main__":
    main()
