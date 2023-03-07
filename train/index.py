from flask import Flask
from flask import request
from flask import render_template
import training
import poly
import os
app = Flask(__name__)
 
@app.route('/upload', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		f = request.files['file']
		print(request.files)
		f.save("saved_models/input/model_last.pt.tar")
		print('file uploaded successfully')
	else:
		return ' <form action = "" method = "POST" enctype = "multipart/form-data"><input type = "file" name = "file" /><input type = "submit" value="提交"/></form>'
		#return render_template('upload.html')

	file_name=["saved_models/model_MNIST_Oct.21_14.55.26_mnist/model_last.pt.tar","saved_models/input/model_last.pt.tar"]
	poly.model_add(2,file_name,"cpu")
	result = training.main("configs/mnist_params.yaml","mnist")
	os.remove("saved_models/input/model_last.pt.tar")
	return result
	#return 'Hello World'
 
if __name__ == '__main__':
   app.run('0.0.0.0', port=5000)
