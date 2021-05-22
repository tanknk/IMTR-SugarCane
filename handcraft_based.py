# ไลบรารี่พื้นฐาน
import cv2
import tqdm as t
import numpy as np
from os import listdir
from joblib import dump, load

# ไลบรารี่สำหรับแบบจำลอง KNN
import sklearn.neighbors as sn

# ไลบรารี่สำหรับเมทริกซ์ระดับสีเทาร่วม (GLCM)
import skimage.feature as skf

# ไลบรารี่สำหรับการตรวจสอบความแม่นยำของแบบจำลอง
import sklearn.metrics as sm

# ไลบรารี่สำหรับการแบ่งชุดข้อมูลเป็น 1.ชุดข้อมูลสำหรับฝึกสอน 2.ชุดข้อมูลสำหรับทดสอบ
import sklearn.model_selection as sms

allClass = {    # allClass เป็น Dict ที่มี key เป็นเลข 1-6 โดยมี value เป็นชื่อ Class
    1: 'green',
    2: 'red',
    3: 'ring',
    4: 'spot',
    5: 'white',
    6: 'yellow'
}

def featureExtract(img1, img2, className, featureTr, labelTr):
    """ 
    ฟังก์ชันสำหรับสกัดคุณลักษณะ (Feature) ของรูปภาพ โดยมี input ดังนี้
    img1        : รูปภาพสี
    img2        : รูปภาพระดับสีเทา
    className   : คลาสของรูปภาพดังกล่าว
    featureTr   : ลิสต์ของ Feature
    labelTr     : ลิสต์ของ Label

    และมี Output คือ
    featureTr   : ลิสต์ของ Feature ที่เพิ่ม Feature ที่ทำการสกัดเข้าไป
    labelTr     : ลิสต์ของ Label ที่เพิ่ม Label ของ Feature เข้าไป
    """
    
    """ ดึงคุณลักษณะ สี (Color) จากรูปภาพ """

    # แปลงรูปภาพให้อยู่บนปริภูมิสี HSV
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

    # แปลงข้อมูลจากเมตริกซ์ให้อยู่ในรูปแบบเวกเตอร์ เฉพาะค่า Hue
    img1 = img1[:,:,0].reshape(1,-1)

    # สร้างฮิสโตแกรมจากค่า Hue โดยช่วงของแต่ละ bin มีความกว้าง = 8 โดยจะมีทั้งหมด 33 bin
    hist, _ = np.histogram(img1, bins = np.arange(-0.5, 256, 8))

    # Normalization เพื่อทำให้ Feature สามารถรองรับขนาดของรูปภาพที่แตกต่างกันได้
    histNormalize = hist/np.sum(hist)

    """ ดึงคุณลักษณะ ลวดลาย (Texture) จากรูปภาพ """
    # Image Quantization
    img2 = (img2 / (256/64)).astype(int)

    # สร้าง GLCM โดยตั้งค่า offest เป็น 3 ระดับ ได้แก่ [1,2,3] พร้อมทั้งแบ่งทิศตามการเปลี่ยนแปลงของข้อมูล [0 องศา, 45 องศา, 90 องศา, 135 องศา]
    # พร้อมทั้งกำหนดระดับสีเทาที่ระดับ 64 
    glcm = skf.greycomatrix(img2, distances=[1, 2, 3], angles=[0, 45, 90, 135], levels=64, symmetric=True, normed=True)

    # สกัดคุณลักษณะ (Feature) จากรูปภาพ
    featureCon = skf.greycoprops(glcm, 'contrast')[0]       # สกัด Contrast จากรูปภาพ 
    featureEne = skf.greycoprops(glcm, 'energy')[0]         # สกัด Energy จากรูปภาพ    
    featureHom = skf.greycoprops(glcm, 'homogeneity')[0]    # สกัด Homogeneity จากรูปภาพ
    featureCor = skf.greycoprops(glcm, 'correlation')[0]    # สกัด Correlation จากรูปภาพ

    # รวม Feature ต่าง ๆ เป็นลิสต์ 1 มิติ
    features = np.hstack((featureCon, featureEne, featureHom, featureCor, histNormalize))

    featureTr.append(features)
    labelTr.append(className)

    return featureTr, labelTr


def train():
    """
    เป็นฟังก์ชันสำหรับการฝึกสอน (Train) แบบจำลองด้วยข้อมูลจากโฟลเดอร์ที่มีโครงสร้างดังนี้
    handcraft_based.py
    |---Tr
        |---green
        |---red
        |---ring
        |---spot
        |---white
        |---yellow
    """

    featureTr = []  # featureTr ใช้เก็บคุณลักษณะ (Feature) ที่สกัดออกมาเพื่อนำไปใช้ในการฝึกสอน
    labelTr = []    # labelTr ใช้เก็บ ชื่อของ Class เพื่อนำไปจับกับค่าใน featureTr
    
    # ดึงรูปภาพจากแต่ละ Class ในโฟลเดอร์สำหรับฝึกสอน
    for classname in t.tqdm(allClass.values()):
        for id in listdir('dataset/Tr/' + classname):
            if id.endswith(".JPG"):
                image1 = cv2.imread('dataset/Tr/' + classname + '/' + id)       # ดึงรูปภาพสี
                image2 = cv2.imread('dataset/Tr/' + classname + '/' + id, 0)    # ดึงรูปภาพระดับสีเทา
                featureExtract(image1, image2, classname, featureTr, labelTr)
                
                # ทำการกลับภาพ (Flip) เพื่อทำ Data augmentation -> เพื่อเพิ่มชุดข้อมูล (dataset)
                image1 = cv2.flip(image1, 1)
                image2 = cv2.flip(image2, 1)
                featureTr, labelTr = featureExtract(image1, image2, classname, featureTr, labelTr)

    # แบ่งชุดข้อมูล เป็น 1. ชุดข้อมูลสำหรับฝึกสอน (feature_train, label_train) 2.ชุดข้อมูลสำหรับทดสอบ (feature_test, label_test)
    feature_train, feature_test, label_train, label_test = sms.train_test_split(featureTr, labelTr, test_size=0.1)

    # สร้าง KNN Classifier
    knn = sn.KNeighborsClassifier(n_neighbors=1, metric='euclidean')

    # ฝึกสอนแบบจำลอง KNN โดยใช้ชุดข้อมูลสำหรับฝึกสอน
    knn.fit(feature_train, label_train)

    # ทำนายผลลัพธ์ (Label, Class) จากข้อมูลคุณลักษณะ (Feature) สำหรับทดสอบ
    knn_pred = knn.predict(feature_test)

    # แสดงค่าความแม่นยำของแบบจำลอง KNN
    print("KNN Accuracy:", sm.accuracy_score(label_test, knn_pred) * 100, '%')

    # แสดงผลลัพธ์ที่คาดหวังและผลลัพธ์จากการทำนายของแบบจำลอง KNN
    print("Result")
    print("Expect: ", label_test)
    print("KNN prediction: ", knn_pred)

    # บันทึกแบบจำลอง KNN (ในกรณีที่ต้องการบันทึกแบบจำลอง KNN)
    dump(knn, 'knn_model.joblib')

def test():
    """
    เป็นฟังก์ชันสำหรับการทดสอบ (Test) แบบจำลองด้วยข้อมูลจากแหล่งอื่น ๆ นอกเหนือจากใน dataset
    
    """
    
    featureTs = []  # featureTs ใช้เก็บคุณลักษณะ (Feature) ที่สกัดออกมาเพื่อนำไปใช้ในการทดสอบ
    labelTs = []    # labelTs ใช้เก็บ ชื่อของ Class เพื่อนำไปจับกับค่าใน featureTs
    
    # ที่อยู่ของรูปภาพที่ใช้สำหรับทดสอบ
    path = "dataset/Tr/yellow/DSC00267.jpg" 

    # ดึงแบบจำลองที่บันทึกไว้มาใช้งาน
    knn_model = load('Classifier/knn_model.joblib')

    # ดึงรูปภาพที่ใช้สำหรับทดสอบมาทำการจำแนกหมวดหมู่
    image1 = cv2.imread(path)       # รูปภาพสี
    image2 = cv2.imread(path, 0)    # รูปภาพระดับสีเทา
    featureExtract(image1, image2, 'Test', featureTs, labelTs)  # สกัดคุณลักษณะ (Feature) จากรูปภาพที่ใช้สำหรับทดสอบ

    # ทำนาย Class ของรูปภาพที่ใช้สำหรับทดสอบ
    knn_pred = knn_model.predict(featureTs)

    # แสดงผลลัพธ์จากการทำนายของแบบจำลอง KNN
    print("KNN prediction: ", *knn_pred)

# train() # ในกรณีที่ต้องการสร้างและฝึกฝนแบบจำลองใหม่
test()