import torch 
import numpy as np
#pattern_tensor: torch.Tensor = torch.load("Trojan_Square_10hid")
pattern_tensor: torch.Tensor = torch.load("Trojan_Square_10hid")
torch.set_printoptions(threshold=np.inf)
#full_image=pattern_tensor[0]+pattern_tensor[1]+pattern_tensor[2]
#print(full_image+30 * (full_image == -30))

weight=1 * (pattern_tensor != -10)

print(weight)