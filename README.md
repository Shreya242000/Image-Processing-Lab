# Image-Processing-Lab


**1. Develop a program to display grayscale image using read and write operation.**<br>
import cv2<br><br>
img=cv2.imread('rose1.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT

![image](https://user-images.githubusercontent.com/97940851/174562233-db89101a-0ca1-462f-b481-ab615dd9b6d4.png)

**2. Develop a program to display the image using matplotlib.**<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('leaf1.jpg')<br>
plt.imshow(img)<br>

OUTPUT

![image](https://user-images.githubusercontent.com/97940851/174561806-3d48847f-425c-48bb-9cca-4767735b8df4.png)



**3. develop a program to perform linear transformation. Rotation**<br>
import cv2<br><br>
from PIL import Image<br><br>
img=Image.open('plant4.jpg')<br><br>
img=img.rotate(180)<br><br>
img.show()<br><br>
cv2.waitKey(0)<br><br>
cv2.destroyAllWindows()<br><br>

OUTPUT

![image](https://user-images.githubusercontent.com/97940851/174562744-81b05612-94af-4eac-81d7-8f1c808e14e2.png)



**4. Develop a program to convert colour string to RGB color values.**<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("Yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>

OUTPUT

(255, 255, 0)
(255, 0, 0)

**5. Write a program to create Image using programs.**<br>
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>

OUTPUT

![image](https://user-images.githubusercontent.com/97940851/174564403-b6637935-61ef-4122-9835-0bf8cf68bfbe.png)


**6. Develop a program to visualize the image using various color space.**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('rose1.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>


OUTPUT

![image](https://user-images.githubusercontent.com/97940851/174563392-3f0f2f6d-8a8f-4b78-a01e-72453d803ad7.png)

![image](https://user-images.githubusercontent.com/97940851/174563435-7e22e7f1-3ca9-4113-b06e-f5a2ac80b5d8.png)

![image](https://user-images.githubusercontent.com/97940851/174563481-46011097-6156-4bdd-af27-fc23ff5a4c0f.png)


**7. Write a program to display the image attributes.**<br>
from PIL import Image<br>
image=Image.open('plant4.jpg')<br>
print("FileName: ",image.filename)<br>
print("Format: ",image.format)<br>
print("Mode: ",image.mode)<br>
print("Size: ",image.size)<br>
print("Width: ",image.width)<br>
print("Height: ",image.height)<br>
image.close();<br>

**OUTPUT**

FileName: plant4.jpg<br>
Format: JPEG
Mode: RGB
Size: (480, 720)
Width: 480
Height: 720


**8.Resize the original image**<br>
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

**9.Convert the original image to grayscale and then to binary**<br>
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

OUTPUT

![image](https://user-images.githubusercontent.com/97940851/174060419-cdbae202-bd2b-48c5-a8d0-5807d12fa105.png)

![image](https://user-images.githubusercontent.com/97940851/174060490-6b7d0a40-504b-4575-8783-5d753a802136.png)

![image](https://user-images.githubusercontent.com/97940851/174060627-e4f005d9-04d5-4b41-9c88-52ddbf7d0138.png)


**url**<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ3VyCF39_x0MhTZema9w9qnFPw6SgAAnY0lA&usqp=CAU'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>

**OUTPUT**

![image](https://user-images.githubusercontent.com/97940851/175006849-d1fe66de-25aa-4979-bc09-6ec770b11e3f.png)


**Write aprogram to mask and blur the image**<br>
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('fish.jpg')<br><br>
plt.imshow(img)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97940851/175256088-534b41d1-7204-4e74-8ffa-b2e07fd3a6a3.png)

hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(18,255,255)<br>
mask=cv2.inRange(img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>
<br>
![image](https://user-images.githubusercontent.com/97940851/175256433-cd6a66bd-0eec-42ed-9281-3a002ad19731.png)


light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97940851/175256779-ede4be6e-d028-445b-8ec0-b748f1f9fe41.png)


final_mask=mask+mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_result)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97940851/175256930-9ae771e4-4cb7-43ca-bf39-68b2dbc07361.png)


blur=cv2.GaussianBlur(final_result,(7,7),0)<br>
plt.imshow(blur)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97940851/175257079-4bb9e433-07be-4e17-a7df-43fbac0010f6.png)


**Write a program to perform airthmetic operation on image**

import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>

#Read image file<br>
img1=cv2.imread('image1.jpg')<br>
img2=cv2.imread('image2.jpg')<br>

#numpy addition on image<br>
fimg1 = img1 + img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>

#saving the output image<br>
cv2.imwrite('output.jpg',fimg1)<br>
fimg2 = img1 - img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>
#saving<br><br>
cv2.imwrite('output.jpg',fimg2)<br>
fimg3 = img1 * img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>
#saving<br>
cv2.imwrite('output.jpg',fimg3)<br>
fimg4 = img1 / img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>
#saving<br>
cv2.imwrite('output.jpg',fimg4)<br>

**OUTPUT**

![image](https://user-images.githubusercontent.com/97940851/175271361-f69fd056-0c9f-48de-9f0e-58f179e3165a.png)

![image](https://user-images.githubusercontent.com/97940851/175271547-755682e2-fb06-415c-becb-c9dfc61547b8.png)

![image](https://user-images.githubusercontent.com/97940851/175271715-3e037c18-5be8-4457-9ce2-2f77386b4f49.png)

![image](https://user-images.githubusercontent.com/97940851/175272011-17225891-481f-445d-b47e-1fb7804c5997.png)



****
import cv2<br>
img=cv2.imread("dog.jpg")<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br><br>

**OUTPUT**

![image](https://user-images.githubusercontent.com/97940851/175275459-1038aabc-f2e2-48d4-832e-4818794576d7.png)

![image](https://user-images.githubusercontent.com/97940851/175275565-e98518fb-4310-4ae7-9121-312f769a1789.png)

![image](https://user-images.githubusercontent.com/97940851/175275777-13baace2-7b44-40e2-8fca-601994e1dd4f.png)

![image](https://user-images.githubusercontent.com/97940851/175275902-3fdd3a34-d608-4f77-85c8-f3164238cb19.png)

![image](https://user-images.githubusercontent.com/97940851/175275976-0d96f5e4-beb2-4d92-b5b6-714205902ea2.png)



**Program to create an image using**
import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array=np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100]=[255,130,0]<br>
array[:,100:]=[0,0,255]<br>
img=Image.fromarray(array)<br>
img.save('IMAGES.jpg')<br>
img.show()<br>
c.waitKey(0)<br><br>

**output**

![image](https://user-images.githubusercontent.com/97940851/175283707-b07903e6-eeae-4838-85e7-bd7f7177f98c.png)


**Bitwise Operator**
import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('images1.jpg',1)<br>
image2=cv2.imread('images1.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr=cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2<br>
cv2.waitKey(0)<br>

**OUTPUT**

![image](https://user-images.githubusercontent.com/97940851/176408534-33a0ecb4-97a0-4532-bf63-8a862d95fc5f.png)


**Blurring Image**

#importing libraries<br>
import cv2<br>
import numpy as np<br>
image=cv2.imread('image1.jpg')<br>
cv2.imshow('Original Image',image)<br>
cv2.waitKey(0)<br>
#GaussianBlur<br>
Gaussian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow('Gaussian Blurring',Gaussian)<br>
cv2.waitKey(0)<br>
#Median blur<br>
median=cv2.medianBlur(image,5)<br>
cv2.imshow('Median Blurring',median)<br>
cv2.waitKey(0)<br>
#Bilateral Blur<br>
bilateral=cv2.bilateralFilter(image,9,75,75)<br>
cv2.imshow('Bilateral Blurring',bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**OUTPUT**

![image](https://user-images.githubusercontent.com/97940851/176413343-d96aca82-6cf8-4db3-9b18-03ff4761fbb5.png)

![image](https://user-images.githubusercontent.com/97940851/176413421-57c3c874-ba57-4362-ab42-f4ea3b23339a.png)

![image](https://user-images.githubusercontent.com/97940851/176413499-f6c02fcd-98d0-481a-be6d-b9a4f0dab7d0.png)

![image](https://user-images.githubusercontent.com/97940851/176413558-645ccdeb-054a-4ae3-8b9d-581ece3760dd.png)



**Image Enhancement**

from PIL import Image<br>
from PIL import ImageEnhance<br>
image=Image.open('lotus.jpg')<br>
image.show()<br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri.enhance(brightness)<br>
image_brightened.show()<br>
enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>
image_contrasted=enh_con.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha=ImageEnhance.Sharpness(image)<br>
sharpness=3.0<br>
image_sharped=enh_sha.enhance(sharpness)<br>
image_sharped.show()<br>

**OUTPUT**

![image](https://user-images.githubusercontent.com/97940851/176418038-15ffb777-77a4-4655-9777-fa2de9d0da32.png)

![image](https://user-images.githubusercontent.com/97940851/176418104-374c887d-050d-45a6-92c1-1341de02ba3f.png)

![image](https://user-images.githubusercontent.com/97940851/176418174-de365094-1245-430f-8e52-0794dc9f70d5.png)

![image](https://user-images.githubusercontent.com/97940851/176418294-f6ba9fe4-8f7e-449e-bb45-c25657b2f1d7.png)

![image](https://user-images.githubusercontent.com/97940851/176418351-7d56cbee-457d-4183-8470-8c1325c98cde.png)


**Morphological Operation**

import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img=cv2.imread('img1.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel=np.ones((5,5),np.uint8)<br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)<br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)<br>
erosion=cv2.erode(img,kernel,iterations=1)<br>
dilation=cv2.dilate(img,kernel,iterations=1)<br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>

**OUTPUT**

![image](https://user-images.githubusercontent.com/97940851/176424583-7f0129f9-fd1a-4e78-b3ef-7bdd20a7e17a.png)


**Gray Scale**<br>
import cv2<br>
OriginalImg=cv2.imread('rose.jpg')<br>
GrayImg=cv2.imread('rose.jpg',0)<br>
isSaved=cv2.imwrite('D:\i.jpg',GrayImg)<br>
cv2.imshow('Display Original Image',OriginalImg)<br>
cv2.imshow('Display Grayscale Image',GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
    print('The image is successfully saved.')<br>
    
    
 **OUTPUT**
 
 ![image](https://user-images.githubusercontent.com/97940851/178697677-e6140969-1eef-4c1e-8275-24c56769f317.png)<br>

 ![image](https://user-images.githubusercontent.com/97940851/178697495-bb729e04-5414-4beb-b290-df466f921022.png)


**Graylevel sclicing with background**<br>

import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('lotus.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
                z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

**OUTPUT**

![image](https://user-images.githubusercontent.com/97940851/178703617-9ff09f52-ae4a-4f04-b22b-954c1eae65d4.png)




**Graylevel slicing without background**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('lotus.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
                z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

**OUTPUT**

![image](https://user-images.githubusercontent.com/97940851/178702820-5a431c4b-4224-49b4-b441-ff0ef0162668.png)
<br>


**Histogram**

import numpy as np<br>
import skimage.color<br>
import skimage.io<br>
import matplotlib.pyplot as plt<br>
#matplotlib widget<br>

/read the image of a plant seedling as grayscale from the outset/<br>
image = skimage.io.imread(fname="rose.jpg", as_gray=True)<br>
image1 = skimage.io.imread(fname="rose.jpg")<br>
/display the image/<br>
fig, ax = plt.subplots()<br>
plt.imshow(image, cmap="gray")<br>
plt.show()<br>

fig, ax = plt.subplots()<br>
plt.imshow(image1, cmap="gray")<br>
plt.show()<br>


/create the histogram/<br>
histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))<br>

/configure and draw the histogram figure/<br>
plt.figure()<br>
plt.title("Grayscale Histogram")<br>
plt.xlabel("grayscale value")<br>
plt.ylabel("pixel count")<br>
plt.xlim([0.0, 1.0])  /<- named arguments do not work here/<br>

plt.plot(bin_edges[0:-1], histogram)   /<- or here/<br>
plt.show()<br>

**OUTPUT**
<br>
![image](https://user-images.githubusercontent.com/97940851/178964241-da19198d-d938-49a6-a654-26f54641c6d4.png)

![image](https://user-images.githubusercontent.com/97940851/178964512-84b945d8-3dd4-4fa8-8e1e-de3aca04953c.png)

![image](https://user-images.githubusercontent.com/97940851/178964546-dd08d960-e1ca-4fa4-b2e9-d4bcf43b620b.png)


**Image Negetive**<br>

#matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('bfly.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>

![image](https://user-images.githubusercontent.com/97940851/179959972-0f4f8c90-d3de-4edf-93f9-6c0e96bf3948.png)<br>

negetive=255-pic #neg=(L-1)-img<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negetive);<br>
plt.axis('off');<br><br>

![image](https://user-images.githubusercontent.com/97940851/179960762-a91b2edb-189a-4cc5-942b-43565c44dfb4.png)<br>

#matplotlib inline<br>
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>

pic=imageio.imread('bfly.jpg')<br>
gray=lambda rgb:np.dot(rgb[...,:3],[0.299,0.587,0.114])<br>
gray=gray(pic)<br>

max_=np.max(gray)<br>

def log_transform():<br>
    return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>

![image](https://user-images.githubusercontent.com/97940851/179960918-9bb923e8-a2f0-4ac0-b39d-63c7d4fb12f2.png)<br>

import imageio<br>
import matplotlib.pyplot as plt<br>

#Gamma encoding<br>
pic=imageio.imread('bfly.jpg')<br>
gamma=2.2 #Gamma<1~ Dark;Gamma >1 ~ Bright<br>

gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>

![image](https://user-images.githubusercontent.com/97940851/179961024-d1692aab-6c74-4103-bf24-8fb9b5678360.png)<br>



**Image manipulation**<br>

#Image sharpen<br>
from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>
#load the image<br>
my_image=Image.open('tiger.jpg')<br>
#use sharpen function<br>
sharp=my_image.filter(ImageFilter.SHARPEN)<br>
#save the image<br>
sharp.save('D:\image_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97940851/179961104-47c651e3-6fed-4daa-b7f5-35212d2a2932.png)<br>

#Image flip<br>
import matplotlib.pyplot as plt<br>
#load the image<br>
img=Image.open('tiger.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
#use the flip function<br>
flip =img.transpose(Image.FLIP_LEFT_RIGHT)<br>
#save the image<br>
flip.save('D:/image_flip.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97940851/179961191-1d2d4723-a6ca-4206-ab68-41a6910e8eaa.png)<br>

![image](https://user-images.githubusercontent.com/97940851/179961240-1b46a14c-4fb9-4596-87ce-43deb40f9aaf.png)<br>

#importing Image class from PIL module<br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
#opens a image in RGB mode<br>
im=Image.open('tiger.jpg')<br>

#size of the image in pixels (size of original image)<br>
#(This is not mandatory)<br>
width,height=im.size<br>

#Cropped image of above dimension <br>
#(It will not change original image)<br>
im1=im.crop((50,25,200,160))<br>

#Shows the image in image viewer<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97940851/179961513-06dbf514-5ad2-4a28-8b61-7647ed784f06.png)<br>
























