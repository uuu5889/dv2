# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
import skvideo.io
import cv2
#import cv



import os
import cv2
import numpy as np

col_len=300
row_len=300
duration=500

#import cv2.cv as cv
#path = '/home/m/mycode/cv'
'''path = 'D:\search\sn_resized'
filelist = os.listdir(path)
total_num = len(filelist)


#video=cv2.VideoWriter("VideoTest.avi", cv2.cv.CV_FOURCC('I','4','2','0'), 1, (1280,1024))
#video=cv2.VideoWriter(r"D:\search\sn_resized\video_generized\VideoTest.avi", cv2.VideoWriter_fourcc('I','4','2','0'), 1, (1280,1024))
video=cv2.VideoWriter(r"D:\search\video_generized\VideoTest.avi", cv2.VideoWriter_fourcc('I','4','2','0'), 100, (300,300))
#videoWriter = cv2.VideoWriter('out.avi', cv2.cv.CV_FOURCC('I','4','2','0'), fps, size)
img_index=1
for item in filelist:

 if item.endswith('.jpg'):
     #item='/home/m/mycode/cv/'+item
     item = os.path.join('D:\search\sn_resized',item)
     img = cv2.imread(item)
     print item
     frame=img
     #video.write(img1)

     start_frame = img_index*1000-1000
     # end_frame = int(fps * end)
     end_frame = img_index*1000
     frame_count = img_index*1000-1000
     while frame_count < end_frame:

         #ret, frame = cap.read()
         #frame=img1
         frame_count += 1


         if frame_count >= start_frame:
             video.write(frame)
     print "video generizition finished"
     cv2.imshow("Image", frame)
     key=cv2.waitKey(1000)
     img_index+=1
'''
for i in range(2):
    dir_path=os.path.join(r"D:\search\video_generized",str(i)+'.avi')
    #os.makedirs(dir_path)
    save_path=dir_path
    print 'save_path',save_path
    video = cv2.VideoWriter(save_path,
                            cv2.VideoWriter_fourcc('I', '4', '2', '0'), 100, (300, 300))
    path = os.path.join('D:\search\\',str(i))
    filelist = os.listdir(path)
    print 'filelist',filelist
    total_num = len(filelist)
    img_index=1
    for item in filelist:
     index_sentence=item
     #if item.endswith('.jpg'):
     # item='/home/m/mycode/cv/'+item
     # item = os.path.join('D:\search\sn_resized',item)

     item = os.path.join('D:\search\\', str(i),item,'0.jpg')
     print 'item',item

     img = cv2.imread(item)
     img_index=0
     print 'img', img,np.shape(img)
     while not np.shape(img):
         print 'None'
         img_index+=1
         item = os.path.join('D:\search\\', str(i), index_sentence, str(img_index)+'.jpg')
         print item
         img = cv2.imread(item)
     img = cv2.resize(img, (300, 300))

     frame = img
     # video.write(img1)

     start_frame = img_index * 500 - 500
     # end_frame = int(fps * end)
     end_frame = img_index * 500
     frame_count = img_index * 500 - 500
     while frame_count < end_frame:

         # ret, frame = cap.read()
         # frame=img1
         frame_count += 1

         if frame_count >= start_frame:
             video.write(frame)
     print "video generizition finished"
     cv2.imshow("Image", frame)
     key = cv2.waitKey(1000)
     img_index += 1
    video.release()
    cv2.destroyAllWindows()