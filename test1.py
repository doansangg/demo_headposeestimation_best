import os
import cv2
import sys
sys.path.append('..')
import numpy as np
from math import cos, sin
# from moviepy.editor import *
from lib.FSANET_model import *
import numpy as np
from keras.layers import Average
import time
from detect import *
import imutils
# from moviepy.editor import *
# from mtcnn.mtcnn import MTCNN
def draw(img, rotate_degree):
    for j in range(len(rotate_degree)):
        name = "roll"
        if j == 1:
            name = "pitch"
        if j == 2:
            name = "yaw"
        cv2.putText(img, ('{:05.2f}').format(float(rotate_degree[j])) + "_" + name, (10, 30 + (50 * j)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50):
    print(yaw,roll,pitch)
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
    
def draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot):
    
    # loop over the detections
    if len(detected[0])>2:
        # for i in range(0, detected.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            # confidence = detected[0, 0, i, 2]

            # # filter out weak detections
            # if confidence > 0.5:
            #     # compute the (x, y)-coordinates of the bounding box for
            #     # the face and extract the face ROI
            #     (h0, w0) = input_img.shape[:2]
            #     box = detected[0, 0, i, 3:7] * np.array([w0, h0, w0, h0])
        (startX, startY, endX, endY,_) = detected[0].astype("int")

        # print((startX, startY, endX, endY))
        x1 = startX
        y1 = startY
        w = endX - startX
        h = endY - startY
        
        x2 = x1+w
        y2 = y1+h

        xw1 = max(int(x1 - ad * w), 0)
        yw1 = max(int(y1 - ad * h), 0)
        xw2 = min(int(x2 + ad * w), img_w - 1)
        yw2 = min(int(y2 + ad * h), img_h - 1)
#        cv2.rectangle(input_img, (x1,y1), (x1+w,y1+h), [0,255,0], 2)

        faces[0,:,:,:] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (64, 64))
        # cv2.imshow("image",faces[0])
        # cv2.waitKey(0)
        faces[0,:,:,:] = cv2.normalize(faces[0,:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)        

        # print(faces[0].shape)
        # faces[0]=cv2.resize(faces[0],(64,64))

        face = np.expand_dims(faces[0,:,:,:], axis=0)
        p_result = model.predict(face)
        
        face = face.squeeze()
        img = draw_axis(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], p_result[0][0], p_result[0][1], p_result[0][2])
        rotate_degree=(p_result[0][2], p_result[0][1], p_result[0][0])
        draw(input_img,rotate_degree)
        input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = img
                
    cv2.imshow("result", input_img)
    
    return input_img,rotate_degree #,time_network,time_plot

def main(file):
    # try:
    #     os.mkdir(file[1].replace('\n',''))
    # except OSError:
    #     pass
    # face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
    # detector = MTCNN()
    net, device, cfg = load_net()

    # load model and weights
    img_size = 64
    stage_num = [3,3,3]
    lambda_local = 1
    lambda_d = 1
    img_idx = 0
    detected = '' #make this not local variable
    time_detection = 0
    time_network = 0
    time_plot = 0
    skip_frame = 1 # every 5 frame do 1 detection and network forward propagation
    ad = 0.6

    #Parameters
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    stage_num = [3,3,3]
    lambda_d = 1
    num_classes = 3
    image_size = 64
    num_primcaps = 7*3
    m_dim = 5
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model1 = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    model2 = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    
    num_primcaps = 8*8*3
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model3 = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    
    print('Loading models ...')

    weight_file1 = 'pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    model1.load_weights(weight_file1)
    print('Finished loading model 1.')

    weight_file2 = 'pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    model2.load_weights(weight_file2)
    print('Finished loading model 2.')

    weight_file3 = 'pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model3.load_weights(weight_file3)
    print('Finished loading model 3.')

    inputs = Input(shape=(64,64,3))
    x1 = model1(inputs) #1x1
    x2 = model2(inputs) #var
    x3 = model3(inputs) #w/o
    avg_model = Average()([x1,x2,x3])
    model = Model(inputs=inputs, outputs=avg_model)
    


    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    # protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    # modelPath = os.path.sep.join(["face_detector",
    #     "res10_300x300_ssd_iter_140000.caffemodel"])
    # net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # capture video
# capture video
    cap = cv2.VideoCapture(file[0].replace('\n',''))
    # # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024*1)
    # # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768*1)
    ret, input_img = cap.read()
    #input_img=cv2.rotate(input_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    input_img=cv2.rotate(input_img,cv2.ROTATE_90_CLOCKWISE)
    input_img=imutils.resize(input_img,height=800)
    # img_h, img_w, _ = np.shape(input_img)
    # center=(img_w/2,img_h/2)
    # scale_1=1.0
    # M=cv2.getRotationMatrix2D(center,270,scale_1)
    # input_img=cv2.warpAffine(input_img,M,(img_h,img_w))
    
    frame_width = input_img.shape[1]
    frame_height = input_img.shape[0]
    write_video = True
    if write_video:
       out = cv2.VideoWriter(file[1].replace('\n',''), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))

    # folder_image=file[0].replace('\n','')
    # img_path=os.listdir(folder_image)


    print('Start detecting pose ...')
    # for im in img_path:
    #     image_path=os.path.join(folder_image,im)
    #     img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #     dets, landms = do_detect(img_raw, net, device, cfg)

    #     # get video frame
    #     # ret, input_img = cap.read()
    #     # input_img=cv2.rotate(input_img,cv2.ROTATE_90_CLOCKWISE)
    #     # img_idx = img_idx + 1
    #     img_h, img_w, _ = img_raw.shape

    #     # start_time=time.time()
    #     # if img_idx==1 or img_idx%skip_frame == 0:
    #     #     time_detection = 0
    #     #     time_network = 0
    #     #     time_plot = 0
            
    #     #     # detect faces using LBP detector
    #     #     gray_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    #     #     # detected = face_cascade.detectMultiScale(gray_img, 1.1)
    #     #     # detected = detector.detect_faces(input_img)
    #     #     # pass the blob through the network and obtain the detections and
    #     #     # predictions
    #     #     blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0,
    #     #         (300, 300), (104.0, 177.0, 123.0))
    #     #     net.setInput(blob)
    #     #     detected = net.forward()

    #     #     if detected_pre.shape[2] > 0 and detected.shape[2] == 0:
    #     #         detected = detected_pre

    #     faces = np.empty((1, img_size, img_size, 3))
    #         #print(faces)
    #     input_img,rotate_degree = draw_results_ssd(dets,img_raw,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot)
    #     roll,pitch,yaw=rotate_degree
    #     tet=str(roll)+" "+str(pitch)+" "+str(yaw)+"\n"
    #     cv2.imwrite(file[1].replace('\n','')+"/"+im,input_img)
        # end_time=time.time()
        # fps = 1/(end_time-start_time)
        # fps=int(fps)
        # # cv2.putText(input_img, 'fps: '+('{}').format(fps), (400,100),
        # #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=4, lineType=4)
        # if write_video:
        #     out.write(input_img)
        # else:
        #     input_img = draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot)


        # if detected.shape[2] > detected_pre.shape[2] or img_idx%(skip_frame*3) == 0:
        #     detected_pre = detected

        # key = cv2.waitKey(1)
    yaw=[]
    roll=[]
    pitch=[]
    file2=open(file[2].replace('\n',''),"w")
    while True:
        # get video frame
        ret, input_img = cap.read()
        input_img=cv2.rotate(input_img,cv2.ROTATE_90_CLOCKWISE)
        #input_img=cv2.rotate(input_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        input_img=imutils.resize(input_img,height=800)
        
        #input_img=cv2.rotate(input_img,cv2.ROTATE_270)
        img_idx = img_idx + 1
        img_h, img_w, _ = np.shape(input_img)
        center=(img_w/2,img_h/2)
        scale_1=1.0
        # M=cv2.getRotationMatrix2D(center,270,scale_1)
        # input_img=cv2.warpAffine(input_img,M,(img_h,img_w))
        start_time=time.time()
        if img_idx==1 or img_idx%skip_frame == 0:
            time_detection = 0
            time_network = 0
            time_plot = 0
            
            # detect faces using LBP detector
            # gray_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
            # # detected = face_cascade.detectMultiScale(gray_img, 1.1)
            # # detected = detector.detect_faces(input_img)
            # # pass the blob through the network and obtain the detections and
            # # predictions
            # blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0,
            #     (300, 300), (104.0, 177.0, 123.0))
            # net.setInput(blob)
            # detected = net.forward()

            # if detected_pre.shape[2] > 0 and detected.shape[2] == 0:
            #     detected = detected_pre

            # faces = np.empty((detected.shape[2], img_size, img_size, 3))
            # #print(faces)
            # input_img,rotate_degree = draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot)
            # end_time=time.time()
            # fps = 1/(end_time-start_time)
            # fps=int(fps)
            # print(rotate_degree)
            dets, landms = do_detect(input_img, net, device, cfg)
            faces = np.empty((1, img_size, img_size, 3))
            input_img,rotate_degree = draw_results_ssd(dets,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot)
            if write_video:
                out.write(input_img)
            roll,pitch,yaw=rotate_degree
            tet=str(roll)+" "+str(pitch)+" "+str(yaw)+"\n"
            
            file2.write(tet)
            #cv2.imwrite(file[2].replace('\n','')+"/"+im,input_img)
            # yaw.append(rotate_degree[2])
            # roll.append(rotate_degree[0])
            # pitch.append(rotate_degree[1])
            # cv2.putText(input_img, 'fps: '+('{}').format(fps), (400,100),
            #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=4, lineType=4)

        else:
            input_img = draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot)


        # if detected.shape[2] > detected_pre.shape[2] or img_idx%(skip_frame*3) == 0:
        #     detected_pre = detected

        key = cv2.waitKey(1)
        
if __name__ == '__main__':
    file=open('file.txt',"r")
    path=file.readlines()
    main(path)
