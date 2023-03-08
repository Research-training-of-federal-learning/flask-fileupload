import os
from file import file_exists
from file import file_read
from reverse.MNIST import find_point
import shutil


if __name__ == '__main__':
	os.rmdir("reverse/MNIST/find_result_backup")
	shutil.copytree("reverse/MNIST/find_result","reverse/MNIST/find_result_backup")