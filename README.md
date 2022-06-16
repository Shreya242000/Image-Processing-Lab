# Image-Processing-Lab
**Resize the original image**
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


