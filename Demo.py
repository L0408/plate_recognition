import cv2
import numpy as np
from aip import AipOcr
from PIL import Image, ImageFont, ImageDraw


def paint_chinese_opencv(im,chinese,position,fontsize,color):#opencv输出中文
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))# 图像从OpenCV格式转换成PIL格式
    font = ImageFont.truetype('simhei.ttf',fontsize,encoding="utf-8")
    #color = (255,0,0) # 字体颜色
    #position = (100,100)# 文字输出位置
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,font=font,fill=color)# PIL图片上打印汉字 # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)# PIL图片转cv2 图片
    return img



APP_ID = '17036408'
API_KEY = 'QIrtLEfkKfwjF7XwNRYoGqFL'
SECRET_KEY = 'C1s9K2AS589BwcMwl9rPLxIBEz9fFLtk'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('timg.jpg')
#img = cv2.resize(img, (700, 700))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 2)
# kernel = np.ones((3, 3), np.uint8)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

plats = plate_cascade.detectMultiScale(blur, 2, 4)
for (x, y, w, h) in plats:

    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
try:
	plate = img[y:y+h, x:x+w]


	#cv2.imshow("plate", plate)
	cv2.imshow("img", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


	cv2.imwrite('plate.jpg', plate)

	with open("plate.jpg", "rb") as f:
	     image = f.read()
	     text_lists = client.basicAccurate(image)["words_result"]
	     for text_list in text_lists:
	         print(text_list["words"])

	text = '车牌是：'+text_list['words']
	image = paint_chinese_opencv(plate, text, (0, 0), 20, (255, 0, 0))

	cv2.imshow("image", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

except NameError: 
	print('未识别出车牌.')
