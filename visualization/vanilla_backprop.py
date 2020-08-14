import sys
sys.path.append("..")
import torch
from utils import tensor2cuda, one_hot

class VanillaBackprop():

    def __init__(self, model):
        self.model = model

    def generate_gradients(self, input_image, target_class, bnid=None):
        self.model.eval()
        x = input_image.clone()
        x.requires_grad = True
        with torch.enable_grad():
            if bnid:  
                model_output = self.model(x, bnid)
            else:
                model_output = self.model(x)
            self.model.zero_grad()
            # grad_outputs = one_hot(target_class, model_output.shape[1])
            # grad_outputs = tensor2cuda(grad_outputs)
            grad = torch.autograd.grad(model_output, x, grad_outputs=target_class, 
                        only_inputs=True)[0]
            self.model.train()
        return grad
