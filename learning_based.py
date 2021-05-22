# ไลบรารี่พื้นฐาน
import numpy as np
import cv2
import tqdm as t
from Net import Net
from os import listdir

# ไลบรารี่สำหรับการอ่านและแสดงผลรูปภาพ
from skimage.io import imread

# ไลบรารี่สำหรับการแบ่งชุดข้อมูลเป็น 1.ชุดข้อมูลสำหรับฝึกสอน 2.ชุดข้อมูลสำหรับทดสอบ
import sklearn.model_selection as sms

# ไลบรารี่สำหรับการตรวจสอบความแม่นยำของแบบจำลอง
import sklearn.metrics as sm

# ไลบรารี่สำหรับการใช้งานโมดูลต่าง ๆ ของ PyTorch
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

list_name = ['green', 'red', 'ring', 'spot', 'white', 'yellow'] # list_name เป็น List ที่ใช้สำหรับข้ารหัสชื่อ Class เป็นตัวเลข

def train():
    """
    ฟังก์ชันสำหรับการฝึกสอนแบบจำลอง
    """
    
    featureTr = []  # featureTr ใช้เก็บคุณลักษณะ (Feature) ที่สกัดออกมาเพื่อนำไปใช้ในการฝึกสอน
    labelTr = []  # labelTr ใช้เก็บ ชื่อของ Class เพื่อนำไปจับกับค่าใน featureTr
    
    """ โหลดข้อมูลรูปภาพ """
    for classname in t.tqdm(list_name):
        for id in listdir('dataset/Tr/' + classname):
            if id.endswith(".JPG"):
                # โหลดข้อมูลรูปภาพเข้าสู่โปรแกรม
                img = imread('dataset/Tr/' + classname + '/' + id, as_gray=True)
                
                # กลับ (Flip) รูปภาพสำหรับทำ Data Augmentaion -> เพื่อเพิ่มชุดข้อมูล (dataset)
                img1 = cv2.flip(img, 1)

                # ปรับขนาดรูปภาพให้มีขนาด ... x ...
                width = 512
                height = 512
                
                img = np.resize(img, (width, height, 1))
                img1 = np.resize(img1, (width, height, 1))

                # Normalize ข้อมูลพิกเซลให้อยู่ในช่วง 0 - 1
                img /= 255.0
                img1 /= 255.0

                # แปลงประเภทของรูปภาพให้เป็น float32
                img = img.astype('float32')
                img1 = img1.astype('float32')
                
                # เพิ่มรูปภาพและ เพิ่ม class เข้า list สำหรับการฝึกสอน
                featureTr.append(img)
                labelTr.append(classname)
                print(classname)

                featureTr.append(img1)
                labelTr.append(classname)

    # เข้ารหัสชื่อ Class เป็นตัวเลข
    labelTr = list(map(lambda x: list_name.index(x), labelTr))

    # แปลง featureTr และ labelTr ให้อยู่ในรูปลิสต์ของ Numpy
    featureTr = np.array(featureTr)
    labelTr = np.array(labelTr)
    
    """ การเตรียมข้อมูลชุดฝึกสอนและข้อมูลชุดทดสอบ """
    # แบ่งชุดข้อมูล เป็น 1.ชุดข้อมูลสำหรับฝึกสอน (train_x, train_y) 2.ชุดข้อมูลสำหรับทดสอบ (test_x, test_y)
    train_x, test_x, train_y, test_y = sms.train_test_split(featureTr, labelTr, test_size = 0.1) 
    
    # แปลงชุดข้อมูลสำหรับฝึกสอนให้อยู่ในรูปแบบ torch.float32 เพื่อให้สามารถทำงไนร่วมับ PyTorch ได้
    train_x = train_x.reshape(108 ,1, width, height)
    train_x = torch.from_numpy(train_x).to(torch.float32)
    train_y = train_y.astype(int)
    train_y = torch.from_numpy(train_y).to(torch.float32)

    # แปลงชุดข้อมูลสำหรับทดสอบให้อยู่ในรูปแบบ torch.float32 เพื่อให้สามารถทำงานร่วมกับ PyTorch ได้
    test_x = test_x.reshape(12, 1, width, height)
    test_x = torch.from_numpy(test_x).to(torch.float32)
    test_y = test_y.astype(int)
    test_y = torch.from_numpy(test_y).to(torch.float32)


    """ กำหนดอัลกอลิทึมสำหรับการเรียนรู้ """
    # นำเข้าแบบจำลองจากไฟล์ Net.py
    model = Net()
    
    # กำหนด optimizer
    optimizer = SGD(model.parameters(), lr=0.07, momentum=0.9)
    
    # กำหนดฟังก์ชัน loss
    criterion = CrossEntropyLoss()

    """ การฝีกสอนแบบจำลอง """
    # สร้าง List สำหรับเก็บ loss
    train_losses = []
    
    # กำหนดจำนวนรอบการฝึกสอน
    n_epochs = 30

    # ฝึกสอนแบบจำลอง
    for epoch in t.tqdm(range(n_epochs)):
        model.train()
        tr_loss = 0
        
        # นำเข้าชุดข้อมูลสำหรับฝึกสอน
        x_train, y_train = Variable(train_x), Variable(train_y)
        
        # ล้างค่า optimizer
        optimizer.zero_grad()

        # การทำนายผลลัพธ์ของชุดข้อมูลสำหรับฝึกสอน
        output_train = model(x_train)

        
        # แปลงให้อยู่ในรูปแบบที่เหมาะสม
        y_train = y_train.long()
        y_train = y_train.squeeze_()
    

        """ ประมาณการประสิทธิภาพจากการฝึกสอน """
        loss_train = criterion(output_train, y_train)
        train_losses.append(loss_train)

        # คำนวณน้ำหนักในการอัปเดตของแบบจำลองทั้งหมด
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()


    # บันทึกแบบจำลอง
    # โดยปกติแล้วจะบันทึกอยู่ในรูปแบบ .pt หรือ .pth
    # torch.save(model, 'cnn_model.pt')
    
    with torch.no_grad():
        output = model(test_x)
    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    # แสดงค่าความแม่นยำของแบบจำลอง
    print("Accuracy:", sm.accuracy_score(test_y, predictions) * 100, '%')

def test():
    """
    ฟังก์ชันสำหรับการทดสอบแบบจำลอง
    """

    featureTs = [] # featureTs ใช้เก็บคุณลักษณะ (Feature) ที่สกัดออกมาเพื่อนำไปใช้ในการทดสอบ

    """ โหลดแบบจำลอง """
    # โหลดแบบจำลองจากโฟลเดอร์ Model ที่มีชื่อว่า 'cnn_model.pt'
    model = torch.load('model/cnn_model.pt')
    
    """ โหลดข้อมูลรูปภาพ """
    # กำหนดที่อยู่ของรูปภาพ
    image_path = 'dataset/Tr/yellow/DSC00097.JPG'
    
    # โหลดข้อมูลรูปภาพเข้าสู่โปรแกรม
    img = imread(image_path, as_gray=True)

    # ปรับขนาดรูปภาพให้มีขนาด ... x ...
    img = np.resize(img, (512, 512, 1))

    # Normalize ข้อมูลพิกเซลให้อยู่ในช่วง 0 - 1
    img /= 255.0

    # แปลงประเภทของรูปภาพให้เป็น float32
    img = img.astype('float32')

    # เพิ่มรูปภาพเข้า List สำหรับการทดสอบ
    featureTs.append(img)

    # แปลง featureTs ให้อยู่ในรูปลิสต์ของ Numpy
    featureTs = np.array(featureTs)
    
    # แปลงข้อมูลฝึกสอนให้อยู่ในรูปแบบ torch.float32 เพื่อให้สามารถทำงานร่วมกับ PyTorch ได้
    featureTs = featureTs.reshape(1, 1, 512, 512)
    featureTs = torch.from_numpy(featureTs).to(torch.float32)
    
    """ ทดสอบการจำแนกหมวดหมู่ของรูปภาพด้วยแบบจำลอง """
    # สร้างการคาดการณ์สำหรับชุดทดสอบ
    with torch.no_grad():
        output = model(featureTs)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    
    print("Prediction:" , *list(map(lambda x:  list_name[x], predictions)))
    print(predictions)


# train()
test()