# -*- coding:utf-8 -*-
'''
Author:xiaoufei
Date:2017/10/17
Description: read in a video file, create a display window,and draw user defined text on the window
File: testVideo.py
'''
import numpy as np
import cv2
from PIL import Image,ImageDraw, ImageFont
import os
import ipdb

char_size=20
row_len=300
col_len=400
#读csv文件
import pandas as pd

#加英文字幕
'''
#path = os.getcwd()+'\\局座哭了.csv'
path = r'D:\search\test.csv'
#f = open(path, encoding='utf-8')
f = open(path)
data = pd.read_csv(f)

#print(data)
#print(data.ix[:,'subtitiles'])
#print(data.loc[0,:])

#print(data.loc[:,'subtitles']),type(data.loc[:,'subtitles'])
print(data.loc[0,'subtitles']),type(data.loc[0,'subtitles'])

if __name__ == "__main__":
    # step1: load in the video file
    #videoCapture = cv2.VideoCapture('test.MOV')
    #videoCapture = cv2.VideoCapture(r'D:\search\test.avi')
    videoCapture = cv2.VideoCapture(r'D:\search\video_generized\VideoTest.avi')

    # step2:get a frame
    sucess, frame = videoCapture.read()
    cap = videoCapture
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = int(cap.get(cv2.VideoWriter_fourcc(*'XVID')))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(r'D:\search\video_generized\subt_VideoTest.avi', fourcc, fps, (w, h))

    # step3:get frames in a loop and do process
    
    str = 'sn'
    i=-1
    while (sucess):
        i=i+1
        #pre_frame=[0]
        #if i !=0:
        #    pre_frame = frame[0][0][0]

        pre_frame = frame
        sucess, frame = videoCapture.read()

        if sucess is False:
            break

        #print 'frame',frame,np.shape(frame),type(frame)
        #print 'sucess',sucess
        #displayImg = cv2.resize(frame, (1024, 768))  # resize it to (1024,768)
        displayImg = cv2.resize(frame, (300, 300))

        if i<1000:

            #cv2.putText(displayImg, "Hello World!", (400, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            print str
            cv2.putText(displayImg, str, (200, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        else:
            str='pptv'
            cv2.putText(displayImg, "pptv", (200, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        #cv2.putText(displayImg,data.loc[i,'subtitles'], (400, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        out.write(displayImg)
        cv2.namedWindow('test Video')
        cv2.imshow("test Video", displayImg)

        keycode = cv2.waitKey(1)
        #if keycode == 27:
        #    cv2.destroyWindow('test Video')
        #    videoCapture.release()
        #    break
    print i
    cv2.destroyAllWindows()
    videoCapture.release()
    out.release()
'''

#加不换行的中文字幕
'''if __name__ == "__main__":
# step1: load in the video file
# videoCapture = cv2.VideoCapture('test.MOV')
# videoCapture = cv2.VideoCapture(r'D:\search\test.avi')

# step3:get frames in a loop and do process
    n=0#line number
    text_index=0
    max_text_index=2
    with open(r'D:\search\abstract.txt','r')as f:
        sentence = f.readlines()
        #print 'sentence', sentence
        while(1):

            sentence[n]=sentence[n].decode('utf8')
            #print 'sentence[n]',sentence[n][:1]
            if text_index==max_text_index:
                break
            print 'a new text begins generizing its video'
            #print 'text_index',text_index
            video_path=os.path.join(r'D:\search\video_generized',str(text_index) + '.avi')
            #print 'video_path',video_path #
            videoCapture = cv2.VideoCapture(video_path)

            # step2:get a frame
            sucess, frame = videoCapture.read()
            cap = videoCapture
            frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print 'frame',frame
            print 'a  sentence of this text begins',sentence[n]
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = int(cap.get(cv2.VideoWriter_fourcc(*'XVID')))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(os.path.join(r'D:\search\video_generized','titled_'+str(text_index)+'.avi'), fourcc, fps, (w, h))
            i=-1 #frame number
            while (sucess):
                #print sucess
                i = i + 1
                # pre_frame=[0]
                # if i !=0:
                #    pre_frame = frame[0][0][0]

                pre_frame = frame
                sucess, frame = videoCapture.read()

                if sucess is False:
                    n=n+2
                    print 'this text finished'
                    print 'present frame number is ', i
                    text_index = text_index + 1
                    print 'the next text number is ', text_index
                    break

                # print 'frame',frame,np.shape(frame),type(frame)
                # print 'sucess',sucess
                # displayImg = cv2.resize(frame, (1024, 768))  # resize it to (1024,768)
                displayImg = cv2.resize(frame, (300, 300))

                if i < 500:

                    # cv2.putText(displayImg, "Hello World!", (400, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    #print str[n]
                    #print 'n',n
                    #cv2.putText(displayImg, sentence[n][:1], (200, 50), cv2.FONT_HERSHEY_PLAIN, 0.05, (0, 0, 255), 2)
                    # add chineise subtitles
                    cv2_im = cv2.cvtColor(displayImg, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
                    pil_im = Image.fromarray(cv2_im)

                    draw = ImageDraw.Draw(pil_im)  # 括号中为需要打印的canvas，这里就是在图片上直接打印
                    # font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8") #第一个参数为字体文件路径，第二个为字体大小
                    font = ImageFont.truetype(r"D:/simhei.ttf", 20, encoding="utf-8")  # 第一个参数为字体文件路径，第二个为字体大小

                    draw.text((0, 0), sentence[n], (0, 0, 255), font=font)  # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体

                    import numpy as np

                    cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                    #print 'frame adding finished '
                else:

                    print 'the next frame number is ', i
                    print 'a  sentence of this text begins:',sentence[n+1]
                    i=i-500-1
                    n=n+1

                    print 'file end'
                    #cv2.putText(displayImg, sentence[n], (200, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    #cv2.putText(displayImg, sentence[n][:1], (200, 50), cv2.FONT_HERSHEY_PLAIN, 0.05, (0, 0, 255), 2)
                    #add chineise subtitles
                    cv2_im = cv2.cvtColor(displayImg, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
                    pil_im = Image.fromarray(cv2_im)

                    draw = ImageDraw.Draw(pil_im)  # 括号中为需要打印的canvas，这里就是在图片上直接打印
                    # font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8") #第一个参数为字体文件路径，第二个为字体大小
                    font = ImageFont.truetype(r"D:/simhei.ttf", 20, encoding="utf-8")  # 第一个参数为字体文件路径，第二个为字体大小
                    sentence[n] = sentence[n].decode('utf8')
                    draw.text((0, 0), sentence[n], (0, 0, 255), font=font)  # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体

                    import numpy as np

                    cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                    # print 'frame adding finished '

                #print 'i',i


                # cv2.putText(displayImg,data.loc[i,'subtitles'], (400, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                #out.write(displayImg)
                out.write(cv2_text_im)
                cv2.namedWindow('test Video')
                #cv2.imshow("test Video", displayImg)
                cv2.imshow("test Video", cv2_text_im)

                keycode = cv2.waitKey(1)
                #if keycode == 27:
                #    cv2.destroyWindow('test Video')
                #    videoCapture.release()
                #    break

            cv2.destroyAllWindows()
            videoCapture.release()
            out.release()'''

if __name__ == "__main__":
    # step1: load in the video file
    # videoCapture = cv2.VideoCapture('test.MOV')
    # videoCapture = cv2.VideoCapture(r'D:\search\test.avi')

    # step3:get frames in a loop and do process
    n = 0  # line number
    text_index = 0
    max_text_index = 2
    with open(r'D:\search\abstract.txt', 'r')as f:
        sentence = f.readlines()
        # print 'sentence', sentence
        while (1):

            sentence[n] = sentence[n].decode('utf8')
            # print 'sentence[n]',sentence[n][:1]
            if text_index == max_text_index:
                break
            print 'a new text begins generizing its video'
            # print 'text_index',text_index
            video_path = os.path.join(r'D:\search\video_generized', str(text_index) + '.avi')
            # print 'video_path',video_path #
            videoCapture = cv2.VideoCapture(video_path)

            # step2:get a frame
            sucess, frame = videoCapture.read()
            cap = videoCapture
            frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print 'frame', frame
            print 'a  sentence of this text begins', sentence[n]
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = int(cap.get(cv2.VideoWriter_fourcc(*'XVID')))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(
                os.path.join(r'D:\search\video_generized', 'titled_' + str(text_index) + '.avi'),
                fourcc, fps, (w, h))
            i = -1  # frame number
            while (sucess):
                # print sucess
                i = i + 1
                # pre_frame=[0]
                # if i !=0:
                #    pre_frame = frame[0][0][0]

                pre_frame = frame
                sucess, frame = videoCapture.read()

                if sucess is False:
                    n = n + 2
                    print 'this text finished'
                    print 'present frame number is ', i
                    text_index = text_index + 1
                    print 'the next text number is ', text_index
                    break

                # print 'frame',frame,np.shape(frame),type(frame)
                # print 'sucess',sucess
                # displayImg = cv2.resize(frame, (1024, 768))  # resize it to (1024,768)
                displayImg = cv2.resize(frame, (col_len, row_len))

                if i < 500:

                    # cv2.putText(displayImg, "Hello World!", (400, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    # print str[n]
                    # print 'n',n
                    # cv2.putText(displayImg, sentence[n][:1], (200, 50), cv2.FONT_HERSHEY_PLAIN, 0.05, (0, 0, 255), 2)
                    # add chineise subtitles
                    cv2_im = cv2.cvtColor(displayImg, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
                    pil_im = Image.fromarray(cv2_im)

                    draw = ImageDraw.Draw(pil_im)  # 括号中为需要打印的canvas，这里就是在图片上直接打印
                    # font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8") #第一个参数为字体文件路径，第二个为字体大小
                    font = ImageFont.truetype(r"D:/simhei.ttf", char_size,
                                              encoding="utf-8")  # 第一个参数为字体文件路径，第二个为字体大小
                    draw.text((0, 0), '    ', (0, 0, 255),font=font)
                    col_n=40
                    row_n=0
                    for char_n in range(len(sentence[n])):
                        if col_n<col_len:
                            draw.text((col_n, row_n), sentence[n][char_n], (0, 0, 255),font=font)
                            col_n+=char_size
                        else:
                            row_n+=char_size
                            col_n=char_size
                            draw.text((0,row_n),sentence[n][char_n],(0,0,255),font=font)
                    import numpy as np

                    cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                    # print 'frame adding finished '
                else:

                    print 'the next frame number is ', i
                    print 'a  sentence of this text begins:', sentence[n + 1]
                    i = i - 500 - 1
                    n = n + 1

                    print 'file end'
                    # cv2.putText(displayImg, sentence[n], (200, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    # cv2.putText(displayImg, sentence[n][:1], (200, 50), cv2.FONT_HERSHEY_PLAIN, 0.05, (0, 0, 255), 2)
                    # add chineise subtitles
                    cv2_im = cv2.cvtColor(displayImg, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
                    pil_im = Image.fromarray(cv2_im)

                    draw = ImageDraw.Draw(pil_im)  # 括号中为需要打印的canvas，这里就是在图片上直接打印
                    # font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8") #第一个参数为字体文件路径，第二个为字体大小
                    font = ImageFont.truetype(r"D:/simhei.ttf", char_size,
                                              encoding="utf-8")  # 第一个参数为字体文件路径，第二个为字体大小
                    sentence[n] = sentence[n].decode('utf8')
                    #draw.text((0, 0),'    '+ sentence[n], (0, 0, 255),
                    #          font=font)  # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体
                    col_n = 40
                    row_n = 0
                    for char_n in range(len(sentence[n])):
                        if col_n < col_len:
                            draw.text((col_n, row_n), sentence[n][char_n], (0, 0, 255), font=font)
                            col_n += char_size
                        else:
                            row_n += char_size
                            col_n = char_size
                            draw.text((0, row_n), sentence[n][char_n], (0, 0, 255), font=font)
                    import numpy as np

                    cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                    # print 'frame adding finished '

                # print 'i',i

                # cv2.putText(displayImg,data.loc[i,'subtitles'], (400, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                # out.write(displayImg)
                out.write(cv2_text_im)
                cv2.namedWindow('test Video')
                # cv2.imshow("test Video", displayImg)
                cv2.imshow("test Video", cv2_text_im)

                keycode = cv2.waitKey(1)
                '''if keycode == 27:
                    cv2.destroyWindow('test Video')
                    videoCapture.release()
                    break'''

            cv2.destroyAllWindows()
            videoCapture.release()
            out.release()
