import os

def del_file(path):
	ls = os.listdir(path)
	for i in ls:
		c_path = os.path.join(path, i)
		if os.path.isdir(c_path):#如果是文件夹那么递归调用一下
			del_file(c_path)
		else:					#如果是一个文件那么直接删除
			os.remove(c_path)