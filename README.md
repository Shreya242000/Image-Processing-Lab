# Image-Processing-Lab


1. Develop a program to display grayscale image using read and write operation.
import cv2
img=cv2.imread('rose1.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
OUTPUT
![image](https://user-images.githubusercontent.com/97940851/174562233-db89101a-0ca1-462f-b481-ab615dd9b6d4.png)





2. Develop a program to display the image using matplotlib.
import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('leaf1.jpg')
plt.imshow(img)
OUTPUT
![image](https://user-images.githubusercontent.com/97940851/174561806-3d48847f-425c-48bb-9cca-4767735b8df4.png)



3. develop a program to perform linear transformation. Rotation
import cv2
from PIL import Image
img=Image.open('plant4.jpg')
img=img.rotate(180)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
OUTPUT
![image](https://user-images.githubusercontent.com/97940851/174562744-81b05612-94af-4eac-81d7-8f1c808e14e2.png)



4. Develop a program to convert colour string to RGB color values.
from PIL import ImageColor<br>
img1=ImageColor.getrgb("Yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>
OUTPUT
(255, 255, 0)
(255, 0, 0)

5. Write a program to create Image using programs.
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>
OUTPUT
<br>

6. Develop a program to visualize the image using various color space.
import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('rose1.jpg')
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()


OUTPUT
![image](https://user-images.githubusercontent.com/97940851/174563392-3f0f2f6d-8a8f-4b78-a01e-72453d803ad7.png)

![image](https://user-images.githubusercontent.com/97940851/174563435-7e22e7f1-3ca9-4113-b06e-f5a2ac80b5d8.png)

![image](https://user-images.githubusercontent.com/97940851/174563481-46011097-6156-4bdd-af27-fc23ff5a4c0f.png)


7. Write a program to display the image attributes.
from PIL import Image<br>
image=Image.open('plant4.jpg')<br>
print("FileName: ",image.filename)<br>
print("Format: ",image.format)<br>
print("Mode: ",image.mode)<br>
print("Size: ",image.size)<br>
print("Width: ",image.width)<br>
print("Height: ",image.height)<br>
image.close();<br>
OUTPUT
FileName: plant4.jpg<br>
Format: JPEG
Mode: RGB
Size: (480, 720)
Width: 480
Height: 720


**8.Resize the original image**
import cv2<br>
img=cv2.imread('flower.jpg')<br>
print('original image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
cv2.waitKey(0)<br>

#to show resized image<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('resized image',imgresize)<br>
print('Resized image length width',imgresize.shape)<br>
cv2.waitKey(0)<br>

**OUTPUT**
original image length width (181, 278, 3)
Resized image length width (160, 150, 3)
-1<br>

![image](https://user-images.githubusercontent.com/97940851/174060079-5a7f7826-4529-4bb8-8db2-5b193d0f979b.png)

![image](https://user-images.githubusercontent.com/97940851/174060142-c655e013-c599-4868-af67-9410b4bb8245.png)

**Convert the original image to grayscale and then to binary**
import cv2<br>
#read the image file<br>
img=cv2.imread('flower.jpg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>

#grayscale<br>
img=cv2.imread('flower.jpg',0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br>

#Binary image<br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**OUTPUT**
![image](https://user-images.githubusercontent.com/97940851/174060419-cdbae202-bd2b-48c5-a8d0-5807d12fa105.png)

![image](https://user-images.githubusercontent.com/97940851/174060490-6b7d0a40-504b-4575-8783-5d753a802136.png)

![image](https://user-images.githubusercontent.com/97940851/174060627-e4f005d9-04d5-4b41-9c88-52ddbf7d0138.png)


