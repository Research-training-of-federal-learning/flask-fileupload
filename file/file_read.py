import os

def file_read(file_name):
	f=open(file_name, encoding='utf-8')
	c=f.read()
	f.close()
	return c
if __name__ == '__main__':
	database="MNIST"
	model="simplenet"
	print(file_read("../pre_models/"+database+"/"+model+"/acc.txt"))