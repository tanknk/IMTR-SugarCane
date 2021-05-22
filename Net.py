import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout

# คลาส Net ใช้สำหรับการกำหนดโครงสร้าง (Architecture) ของ Convolution Neural Network
class Net(Module): 
    def __init__(self): # กำหนดโครงสร้างภายใน Constructor ของคลาส Net
        super(Net, self).__init__()
        """ Feature extraction """
        self.cnn_layers = Sequential(
            # สร้างเลเยอร์ของ 2D Convolution
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(8),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(8, 12, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(12),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(12, 16, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(16, 20, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(20),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(20, 24, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(24),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(24, 28, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(28),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(28, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        """ Classification """
        # สร้างเลเยอร์ของ Linear
        self.linear_layers = Sequential(
            Linear(32 * 2 * 2, 6)
        )   
        
    # กำหนดลำดับการทำงานของโครงสร้าง
    def forward(self, x):
        x = self.cnn_layers(x)
        
        # แปลง Feature map ให้อยู่ในรูปของ Feature vector 
        x = x.view(x.size(0), -1) 
        
        x = self.linear_layers(x)
        return x