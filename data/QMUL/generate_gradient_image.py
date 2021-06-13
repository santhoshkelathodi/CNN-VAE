import cv2
import os
import glob
import numpy as np
import math

image = cv2.imread('qmul_bg.png')
cv2.imshow("qmul_bg",image)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)

#s.fill(255)
#v.fill(255)
hsv_image = cv2.merge([h, s, v])
x_max = image.shape[1] #width
y_max = image.shape[0] #height

print (x_max)
print (y_max)

image[:,:,:] = 0
hsv_image[:,:,:] = 0
print(image.shape)
cv2.waitKey()

src_path = 'Trajectory/'
dest_path = 'image_data/'
classes = ['0'] # only for test
#classes = ['1', '2', '3', '4','5', '6', '7', '8', '9', '10', '11', '12', '13', '14','15', '16', '17', '18', '19', '20']
#classes = ['1', '2', '3', '4','5', '6', '7', '8', '9', '10', '11', '12', '13', '14','15']
#classes = ['overall']
'''
filename = "somefile.txt"  
myfile = open(filename)  
lines = len(myfile.readlines())  
'''
x = []
y = []
t = []
for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
    index = classes.index(fld)
    print('Loading {} files (Index: {})'.format(fld, index))
    path = os.path.join(src_path, fld, '*.txt')#specify the kind of files if * is used all will be added
    #print path
    files = glob.glob(path)
    for fl in files:
        print(classes[index])
	#print(fl)
        with open(fl,"r") as text_file:
            for line in text_file:
		#line = line.translate(string.maketrans("\n\t\r", "   "))
                x_str,y_str,t_str = line.split()
		#x_str,y_str = line.split()
                #print line.index()
                x_val = int(math.floor(float(x_str)))
                y_val = int(math.floor(float(y_str)))
                if (x_val >= x_max):
                    x_val = x_max - 1
                if (y_val >= y_max):
                    y_val = y_max - 1
		#t_val = int(t_str)
		#print x_val           
                x.append(x_val)
                y.append(y_val)
		#t.append(t_val)
	#print x
        num_data = len(x)
        print(num_data)
	#print y
	#print t
        x_coord = np.array(x)
        y_coord = np.array(y)
        x[:] = [val for val in x]
	#t[:] = [x - t[0] for x in t] # subtract from the first value to get offset of time
        t = np.arange(num_data)
        hue = []
        sat = []
        val = []
        if (num_data > 1):
            hue[:] = [x*180.0/t[num_data-1] for x in t]
        else:
            hue[:] = [x*180.0 for x in t]	
	#print t
        sat[:] = [255 for x in t]
        val[:] = [255 for x in t]
	#print sat
	#print val
	#print y[0]
        #put color gradient in the image file
	#((float)time/(float)timeDur)*180;//hue
        hsv_image[y_coord[:],x_coord[:],0] = hue[:]
        hsv_image[y_coord[:],x_coord[:],1] = sat[:]
        hsv_image[y_coord[:],x_coord[:],2] = val[:]

        #for i in range(num_data): 
        #    x_val = x_coord[i]
	#    y_val = y_coord[i]
        #    hsv_image[y_val, x_val, 0] = hue[i]
        #    hsv_image[y_val, x_val, 1] = sat[i]
        #    hsv_image[y_val, x_val, 2] = val[i]
	#cv2.line(img,(0,0),(511,511),(255,0,0),8,8)# can be done only in BGR space
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
	# if not copied the pixels  in the nearhood gets overwritten hence always red color will come for the trajectory
        bgrcopy = image[:, :, :].copy() 
        prev_col = x_coord[0]
        prev_row = y_coord[0]
        print(image[prev_row, prev_col])
        for i in range(1,num_data): 
            b = image[prev_row, prev_col, 0]
            g = image[prev_row, prev_col, 1]
            r = image[prev_row, prev_col, 2]
            cv2.line(bgrcopy,(prev_col, prev_row),(x_coord[i], y_coord[i]), (int(b),int(g),int(r)), 8, 8) # Point(row,col)
            prev_col = x_coord[i]
            prev_row = y_coord[i]
        cv2.imshow('track', bgrcopy)
        # Create target filename
        src_file_name = fl.split('/')[2]
        print(src_file_name)
        destfile = fl.split('/')[1]+"."+src_file_name.split('.')[0]+".jpg"
        destfile = os.path.join(dest_path, fld, destfile) 
        if not os.path.exists(dest_path+fld):
            os.makedirs(dest_path+fld)
        print(dest_path+fld)
        print(destfile)
        cv2.imwrite(destfile, bgrcopy)
	# Prepare for the next file
        x = []
        y = []
        t = []
        hsv_image[:,:,:] = 0 #comment this if you want to see all trajectory

imageWidth = hsv_image.shape[1] #Get image width
imageHeight = hsv_image.shape[0] #Get image height
print(imageWidth)
print(imageHeight)
xPos, yPos = 0, 0
colorHue = 0
print(hsv_image.shape)
'''
hsv_image[:,:,1] = 255
hsv_image[:,:,2] = 255
while xPos < imageWidth: #Loop through rows
    hsv_image[:,xPos,0] = colorHue
    xPos = xPos + 1 #Increment X position by 1
    colorHue = colorHue + 1
    if (colorHue == 180):
	break;
'''
#convert to BGR
image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
cv2.imshow('imagecolorgradient', image) #to display all the trajectories
cv2.imwrite("result.jpg", image) #Write image to file

cv2.waitKey()
