# Image-Processing-Lab


1. Develop a program to display grayscale image using read and write operation.
pip install opencv-python
import cv2
img=cv2.imread('flower5.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
OUTPUT



2. Develop a program to display the image using matplotlib.
import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('plant4.jpg')
plt.imshow(img)
OUTPUT



3. develop a program to perform linear transformation. Rotation
import cv2
from PIL import Image
img=Image.open("plant4.jpg")
img=img.rotate(180)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
OUTPUT



4. Develop a program to convert colour string to RGB color values.
from PIL import ImageColor
img1=ImageColor.getrgb("Yellow")
print(img1)
img2=ImageColor.getrgb("red")
print(img2)
OUTPUT
(255, 255, 0)
(255, 0, 0)

5. Write a program to create Image using programs.
from PIL import Image
img=Image.new('RGB',(200,400),(255,255,0))
img.show()
OUTPUT


6. Develop a program to visualize the image using various color space.
import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('butterfly3.jpg')
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()
OUTPUT

7. Write a program to display the image attributes.
from PIL import Image
image=Image.open('plant4.jpg')
print("FileName: ",image.filename)
print("Format: ",image.format)
print("Mode: ",image.mode)
print("Size: ",image.size)
print("Width: ",image.width)
print("Height: ",image.height)
image.close();
OUTPUT
FileName: plant4.jpg
Format: JPEG
Mode: RGB
Size: (480, 720)
Width: 480
Height: 720


**8.Resize the original image**
import cv2
img=cv2.imread('flower.jpg')
print('original image length width',img.shape)
cv2.imshow('original image',img)
cv2.waitKey(0)

#to show resized image
imgresize=cv2.resize(img,(150,160))
cv2.imshow('resized image',imgresize)
print('Resized image length width',imgresize.shape)
cv2.waitKey(0)

**OUTPUT**
original image length width (181, 278, 3)
Resized image length width (160, 150, 3)
-1

![image](https://user-images.githubusercontent.com/97940851/174060079-5a7f7826-4529-4bb8-8db2-5b193d0f979b.png)

![image](https://user-images.githubusercontent.com/97940851/174060142-c655e013-c599-4868-af67-9410b4bb8245.png)

**Convert the original image to grayscale and then to binary**
import cv2
#read the image file
img=cv2.imread('flower.jpg')
cv2.imshow("RGB",img)
cv2.waitKey(0)

#grayscale
img=cv2.imread('flower.jpg',0)
cv2.imshow("Gray",img)
cv2.waitKey(0)

#Binary image
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("Binary",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

**OUTPUT**
![image](https://user-images.githubusercontent.com/97940851/174060419-cdbae202-bd2b-48c5-a8d0-5807d12fa105.png)

![image](https://user-images.githubusercontent.com/97940851/174060490-6b7d0a40-504b-4575-8783-5d753a802136.png)

![image](https://user-images.githubusercontent.com/97940851/174060627-e4f005d9-04d5-4b41-9c88-52ddbf7d0138.png)


