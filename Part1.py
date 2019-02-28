
# coding: utf-8

# # Import Libraries
# 

# In[12]:


from os import listdir
from os.path import isfile , join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import glob
import cv2


# # A function to show RGB images
# 
# 

# In[13]:


def view( image ):
    plt.figure(figsize=(10,20))
    plt.imshow( image )


# ## 1. Load Images and show them

# In[14]:


images_files = [ join("./images" , f) for f in listdir("images") if isfile(join("images" , f)) ]

images = [ mpimg.imread( f ) for f in images_files ]
color = ('b','g','r')
imageList = list(images)
[view (x) for x in imageList ]


# ## 2. Make Histogram of all three color channels for each image

# In[15]:


for x in range(len (imageList)):
    for i,col in enumerate(color):  
        histr = cv2.calcHist([imageList[x]],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])       
    
    plt.show()


# # Make a function for mouse event

# In[18]:


def mouse_event(event, x, y, flags, param):
    cv2.EVENT_LBUTTONDOWN
     # Clear Screen
    img = cv2.imread(path,-1)
    win=cv2.rectangle(img,(x-13,y-13),(x+13,y+13),(100,50,255),0)	
    position = "x,y: ("+str(x)+","+str(y)+")"
    RGB= "RGB :"+str(img[y,x])
    avg="mean: "+str(np.mean(win))
    var="variance: "+str(np.std(win))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,position, (x-20,y-40),font, 0.6, (230,55,150), 1, cv2.LINE_AA)
    cv2.putText(img,RGB, (x-20,y-20),font, 0.6, (255,25,0), 1, cv2.LINE_AA)
    cv2.putText(img,avg, (x-20,y-60),font, 0.6, (50,0,50), 1, cv2.LINE_AA)
    cv2.putText(img,var, (x-20,y-80),font, 0.6, (50,0,50), 1, cv2.LINE_AA)
    cv2.imshow('original', img)	


# ## 1.Load some-pigeon image to try mouse event

# In[19]:


path='./images/some-pigeon.jpg'
img = cv2.imread(path)
cv2.imshow('original', img)
cv2.setMouseCallback("original", mouse_event)

cv2.waitKey(0)

cv2.destroyAllWindows


# ### Homogeneous because of small changes in values of variance 

# # Gradient

# ## 1.Gradient using for loops

# In[26]:


path='./images/some-pigeon.jpg'
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gx=[img.shape[0],img.shape[1]]
gy=[img.shape[0],img.shape[1]]

for i in range  (img.shape[0]):
    for j in range (img.shape[1]):
        gy=img[i][j]-img[i][j-1]
    gx=img[i][j]-img[i-1][j] 
    #print (gx)    
    gx2=np.power(gx, 2) 
    #print(gx2)
    #print (gy)    
    gy2=np.power(gy, 2) 
    #print(gy2)
    g=np.sqrt(gx2+gy2)
    print(g)


# ## 2.Gradient without for loop

# ### Function for gray scale image

# In[20]:


def rgb2gray(rgb_image):
    return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])


# ### Function for multiview images

# In[21]:


def multi_view( images ):
    images_count = len( images )
    fig = plt.figure(figsize=(10,20))
    for row in range( images_count  ):
        ax1 = fig.add_subplot( images_count , 1 , row + 1)    
        ax1.imshow( images[ row ] )


# In[24]:


path='./images/Pyramids2.jpg'
img = cv2.imread(path)
gray_image =  rgb2gray( img ) 
gray_image_v = np.roll( gray_image , 1 , 0 )
gray_image_h = np.roll( gray_image , 1 , 1 )

gray_image_gv = np.abs( gray_image - gray_image_v )
gray_image_gh = np.abs( gray_image - gray_image_h )

gray_image_gv2= np.power (gray_image_gv, 2 )
gray_image_gh2= np.power (gray_image_gh, 2 )

gradient= np.sqrt( gray_image_gv2  +gray_image_gh2 )

comb = tuple( (gray_image , gray_image_gv, gray_image_gh, gradient) )

multi_view( comb )
print 


# ### Time execution of for loops is longer than with out using them
