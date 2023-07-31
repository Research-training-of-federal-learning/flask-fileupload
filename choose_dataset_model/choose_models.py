import torchvision

def choosemodels(modelname):
	if(modelname=="freestyle"):
		return 0
	model_method=getattr(torchvision.models,modelname)
	model=model_method()
	model.eval()
	return model
if __name__ == '__main__':
	choosemodels("convnext_tiny")
