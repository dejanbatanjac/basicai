class Accuracy():
    'abstract'       
    def reset(self):
        raise NotImplementedError
    
    def set(self, output, targs):  
        raise NotImplementedError
    
    def get(self):
        raise NotImplementedError
    
    def print(self):
        raise NotImplementedError
    
    

class ArgmaxAccuracy():
    'argmax'
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.correct = 0
        self.targets = 0
    
    def set(self, output, targs):  
        output = output.argmax(dim=-1) 
        self.correct += (output==targs).sum().item()
        self.targets += targs.size(0)
    
    def get(self):
        return self.correct/self.targets
    
    def print(self):
        print(self.correct/self.targets)

AA = ArgmaxAccuracy
