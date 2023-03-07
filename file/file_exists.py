import os

def file_exists(file_name):
	return os.path.exists(file_name)

def path_exists(path_name):
	return os.path.exists(path_name)

if __name__ == '__main__':
	file_exists("../")