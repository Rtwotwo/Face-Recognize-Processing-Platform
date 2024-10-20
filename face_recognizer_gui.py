"""
任务: 实个人独立完成一套人脸识别新系统，实现人脸检测、
    人脸关键点检测、人脸对齐、活体检测（眨眼、张嘴、转头等操作）
    人脸特征提取、人脸识别等功能、GUI界面,
    要求每个功能可在画面进行实时展示的关闭
时间: 2024/09/27 - 2024/10/03
作者: Redal
"""
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import cv2 
import queue, math
import threading
from PIL import Image, ImageTk, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd

import dlib
from imutils import face_utils
import mediapipe as mp


"""定义界面显示GUI"""
class FaceRecognitionAPP(tk.Frame):
    def __init__(self, root=None):
        # 初始化GUI界面参数
        super().__init__(root)
        self.root = root
        self.pack()
        self.frame = None
        self.root.title('人脸识别处理平台-Redal')
        self.root.geometry('800x400')
        self.cap = cv2.VideoCapture(0)
        self.frame_queue = queue.Queue(maxsize=10)
        self.video_label = tk.Label(self)
        self.video_label.grid(row=0, column=0, rowspan=1, columnspan=3, sticky='nsew')
        self.processed_image_label = tk.Label(self.root)
        self.processed_image_label.place(x=610, y=60, width=180, height=180)
        self.mtcnn = MTCNN(image_size=400, margin=0, keep_all=False, post_process=True)
        self.mtcnn_recognizer = MTCNN(image_size=160, margin=0, keep_all=True, post_process=False) #人脸识别query模型
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

        self.stop_thread = False
        self.face_detection_active = False  # 控制是否进行人脸检测
        self.facial_feature_extraction_active = False  # 控制是否进行人脸特征提取
        self.face_keypoints_detection_active = False  # 控制是否进行人脸关键点检测
        self.face_alignment_active = False  # 控制是否进行人脸对齐
        self.toggle_face_alignment_crop_image_active = False  # 控制是否进行对齐、裁剪出人脸
        self.toggle_face_liveness_detection_active = False  # 控制是否进行活体检测
        self.face_recognition_active = False  # 控制是否进行人脸识别
        self.toggle_show_multi_keypoints_active = False  # 控制是否显示多种人脸关键点
        self.toggle_dlib_face_recognize_active = False  # 控制是否使用dlib人脸识别
        self.skip_frame_num , self.face_recognized_name= 0, ''  # 控制跳帧数
        self.eye_count_framenum,self.mouth_count_framenum, self.eye_num,self.mouth_num= 0,0,0,0
        self.facial_feature_extraction_name,self.face_embedding = [],[]
        self.facial_feature_csvdata = {}
        
        self.thread = threading.Thread(target=self.update_video_frame)
        self.thread.daemon = True
        self.thread.start()
        self.predictor = dlib.shape_predictor(r".\model\shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()
        # 初始化 MediaPipe 的人脸检测和面部地标检测器
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode = False, max_num_faces= 10,
            refine_landmarks=True, min_detection_confidence=0.3,
            min_tracking_confidence=0.3)
        self.font = ImageFont.truetype(r'font/联想小新潮酷体.ttf', 18, encoding="utf-8")

    def create_buttons(self):
        x0,y0=25,280 # 设置初始的按键(x,y)位置
        interal_x,interal_y=80,40 # 设置按键间距
        def command_cancel_function():
            # 用于取消命令的函数
            self.face_detection_active = False
            self.facial_feature_extraction_active = False
            self.face_keypoints_detection_active = False
            self.face_alignment_active = False
            self.toggle_face_alignment_crop_image_active = False
            self.toggle_face_liveness_detection_active = False
            self.face_recognition_active = False
            self.toggle_show_multi_keypoints_active = False
            self.toggle_dlib_face_recognize_active = False
            text='                   \n'*4
            self.message_label = tk.Label(self.root,text=text,justify=tk.LEFT,wraplength=160,
                                      fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place(x=25, y=50) 
        self.button_common_video_play = tk.Button(root, text=' 功能取消  ',command=command_cancel_function).place(x=25, y=240)
        self.alignment_button = tk.Button(self.root, text=' 对齐/裁剪',command=self.toggle_face_alignment_crop_image).place(x=105, y=240) 
        self.message_label = tk.Label(self.root,text='功能信息显示区',justify=tk.LEFT,wraplength=160,fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place(x=25, y=10)  
        self.processed_image_show = tk.Label(self.root,text='处理图像显示区',justify=tk.LEFT,wraplength=160,fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place(x=625, y=10)
        self.button_clear_show_image = tk.Button(self.root, text=' 清除显示 ',command=self.clear_show_image).place(x=625, y=240)
        self.button_save_show_image = tk.Button(self.root, text=' 保存显示 ',command=self.save_show_image).place(x=705, y=240)
        self.dropdown_box_multi_keypoints_show = ttk.Combobox(self.root, values=['面部轮廓','面部网格','面部线性'], state='readonly', width=10)
        self.dropdown_box_multi_keypoints_show.current(0),self.dropdown_box_multi_keypoints_show.place(x=625,y=280)
        self.dropdown_box_multi_keypoints_show.bind("<<ComboboxSelected>>", self.toggle_show_multi_keypoints)
        self.dropdown_box_multi_dlib_face_recognize = ttk.Combobox(self.root, values=['人脸采集','特征提取','人脸识别'], state='readonly', width=10)
        self.dropdown_box_multi_dlib_face_recognize.current(0), self.dropdown_box_multi_dlib_face_recognize.place(x=625,y=320)
        self.dropdown_box_multi_dlib_face_recognize.bind("<<ComboboxSelected>>", self.toggle_dlib_face_recognize)
        
        self.button_face_feature_extraction = tk.Button(root, text='特征点提取', command=self.toggle_facial_feature_extraction).place(x=x0, y=y0)
        self.button_face_detection = tk.Button(root, text=' 人脸检测 ', command=self.toggle_face_detection).place(x=x0+interal_x, y=y0)
        self.button_face_keypoints_detection = tk.Button(root, text='关键点检测', command=self.toggle_face_keypoints_detection).place(x=x0, y=y0+interal_y)
        self.button_face_alignment = tk.Button(root, text=' 对齐显示 ', command=self.toggle_face_alignment).place(x=x0+interal_x, y=y0+interal_y)
        self.button_face_liveness_detection = tk.Button(root, text=' 活体检测  ', command=self.toggle_face_liveness_detection).place(x=x0, y=y0+interal_y*2)
        self.button_face_recognition = tk.Button(root, text=' 人脸识别 ', command=self.toggle_face_recognition).place(x=x0+interal_x, y=y0+interal_y*2) 

    def toggle_facial_feature_extraction(self): 
        # 控制人脸特征提取开关-self.facial_feature_extraction_active
        self.facial_feature_extraction_active = not self.facial_feature_extraction_active
        if self.facial_feature_extraction_active: text='特征提取:\n请将面部居中并保持独处于画面'
        else: text='                  \n'*4
        self.message_label = tk.Label(self.root,text=text,justify=tk.LEFT,wraplength=160,
                                      fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place(x=25, y=50) 
        
    def toggle_face_detection(self):
        # 控制人脸检测开关-self.face_detection_active
        self.face_detection_active = not self.face_detection_active
        if self.face_detection_active: text='人脸检测:\n可实时检测画面中的多张人脸'  
        else: text='                  \n'*4
        self.message_label = tk.Label(self.root,text=text,justify=tk.LEFT,wraplength=160,
                                      fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place(x=25, y=50) 
        
    def toggle_face_keypoints_detection(self):
        # 控制人脸关键点检测开关-self.face_keypoints_detection_active
        self.face_keypoints_detection_active = not self.face_keypoints_detection_active
        if self.face_keypoints_detection_active: text='关键点检测:\n可实时显示画面中的人脸关键点'  
        else: text='                  \n'*4
        self.message_label = tk.Label(self.root,text=text,justify=tk.LEFT,wraplength=160,
                                      fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place(x=25, y=50)
         
    def toggle_face_alignment(self):
        # 进行对齐显示
        self.face_alignment_active = not self.face_alignment_active
        if self.face_alignment_active: text='对齐显示:\n检测任意倾斜角度的面部绘制边界框'
        elif self.face_alignment_active == False: text='                  \n'*4
        self.message_label = tk.Label(self.root,text=text,justify=tk.LEFT,wraplength=160,
                                      fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place(x=25, y=50)

    def toggle_face_alignment_crop_image(self):
        # 进行人脸对齐、裁剪出人脸
        self.toggle_face_alignment_crop_image_active = not self.toggle_face_alignment_crop_image_active
        if self.toggle_face_alignment_crop_image_active:text='对齐裁剪:\n将人脸对齐并裁剪出人脸'
        else:text='                  \n'*4
        self.message_label = tk.Label(self.root,text=text,justify=tk.LEFT,wraplength=160,
                                      fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place(x=25, y=50)

    def toggle_face_liveness_detection(self):
        # 进行活体检测
        self.toggle_face_liveness_detection_active = not self.toggle_face_liveness_detection_active
        if self.toggle_face_liveness_detection_active: text='活体检测:\n实时检测画面中人脸是否为活体'
        else: text='                  \n'*4
        self.eye_count_framenum,self.mouth_count_framenum, self.eye_num,self.mouth_num= 0,0,0,0 # 重置眼部、嘴部(帧数)计数器
        self.message_label = tk.Label(self.root,text=text,justify=tk.LEFT,wraplength=160,
                                      fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place()
        
    def toggle_face_recognition(self):
        # 进行人脸识别
        self.face_recognition_active = not self.face_recognition_active
        if self.face_recognition_active: text='人脸识别:\n可识别画面中的人脸'
        else: text='                  \n'*4
        self.message_label = tk.Label(self.root,text=text,justify=tk.LEFT,wraplength=160,
                                      fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place(x=25, y=50)
    def toggle_show_multi_keypoints(self):
        # 显示多种人脸关键点
        self.toggle_show_multi_keypoints_active = not self.toggle_show_multi_keypoints_active
        if self.toggle_show_multi_keypoints_active: text='显示多种关键点:\n显示多种人脸关键点,关闭请点击“功能取消”'
        else: text='                  \n'*4
        self.message_label = tk.Label(self.root,text=text,justify=tk.LEFT,wraplength=160,
                                      fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place(x=25, y=50)

    def toggle_dlib_face_recognize(self):
        # 使用dlib人脸识别
        self.toggle_dlib_face_recognize_active = not self.toggle_dlib_face_recognize_active
        if self.toggle_dlib_face_recognize_active: text='人脸识别:\n使用dlib人脸识别,关闭请点击“功能取消”'
        else: text='                  \n'*4
        self.message_label = tk.Label(self.root,text=text,justify=tk.LEFT,wraplength=160,
                                      fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place(x=25, y=50)

    def update_video_frame(self):
        while not self.stop_thread and self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if ret:
                self.frame = cv2.resize(self.frame, (400, 400))
                
                # 1. command_facial_feature_extraction--人脸特征点提取
                if self.facial_feature_extraction_active:        
                    img = Image.fromarray(cv2.resize(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB),(400,400)))
                    def face_in_center(img):
                        # 判断人脸是否单独居中
                        one_face_in_center = False
                        face_center,prob = self.mtcnn.detect(img, landmarks=False)
                        if face_center is not None and face_center.shape[0] == 1:
                            x1, y1, x2, y2 = [int(i) for i in face_center[0]]
                            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                            if math.sqrt((center_x - 200) ** 2 + (center_y - 200) ** 2) < 20:
                                # 到目标区域内部就立即提取特征
                                one_face_in_center = True
                                return one_face_in_center
                        else: return one_face_in_center
                    one_face_in_center = face_in_center(img) # 判断是否单独居中
                    img_rgb = cv2.resize(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB),(400,400))
                    box, _ = self.mtcnn.detect(img_rgb, landmarks=False) # 已经确保画面中仅有一人

                    if box is not None and one_face_in_center:
                        try: 
                            img = ImageTk.PhotoImage(image=img)
                            self.video_label.config(image=img)
                            self.video_label.image = img
                            # 创建临时操作界面
                            tmp_root = tk.Toplevel(self.root)
                            self.center_window(tmp_root,title='特征提取')
                            entry = tk.Entry(tmp_root);entry.pack()
                            def get_entry_text():
                                self.facial_feature_extraction_name,self.face_embedding = [],[] #清空每一循环的数据
                                self.facial_feature_extraction_name.append(entry.get())
                                tmp_root.destroy()
                            tmp_button = tk.Button(tmp_root, text='获取姓名', command=get_entry_text).pack()
                            self.root.wait_window(tmp_root) # 等待用户输入
                            # 将裁剪出的人脸显示出来
                            x1, y1, x2, y2 = [int(i) for i in box[0]]
                            face= self.mtcnn(img_rgb[y1:y2, x1:x2]) # 裁剪出人脸,以便提取特征
                            img_show = Image.fromarray( cv2.resize(img_rgb[y1:y2, x1:x2], (160, 160)) )
                            img_show.save(f'./facebase/{self.facial_feature_extraction_name[-1]}.png') # 保存图片
                            img_show = ImageTk.PhotoImage(image=img_show)
                            self.processed_image_label.config(image=img_show)
                            self.processed_image_label.image = img_show

                            # 提取特征
                            self.face_embedding.append(self.resnet(face.unsqueeze(0)).detach().numpy().flatten().tolist())
                            self.facial_feature_csvdata= {'姓名': self.facial_feature_extraction_name, '特征': self.face_embedding}
                            df = pd.DataFrame(self.facial_feature_csvdata)
                            df['特征']= df['特征'].apply(lambda x: ','.join(map(str, x))) # 转换为字符串格式
                            try:
                                origin_data = pd.read_csv('./mtcnn/facial_feature.csv') 
                                df.to_csv('./mtcnn/facial_feature.csv',mode='a',header=False, index=False)
                            except:
                                df.to_csv('./mtcnn/facial_feature.csv', index=False)
                                
                            tmp_root = tk.Toplevel(self.root)
                            self.center_window(tmp_root,title='特征提取')
                            message = tk.Message(tmp_root, text=self.facial_feature_extraction_name[-1]+'-特征提取成功',width=150).pack()
                            tmp_button = tk.Button(tmp_root, text='关闭', command=lambda:tmp_root.destroy()).pack()
                            tmp_root.wait_window(tmp_root) # 等待用户关闭
                            self.toggle_facial_feature_extraction()
                        except: pass
                    else:
                        img = cv2.resize(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB),(400,400))
                        cv2.circle(img, (200, 200), 150, (255, 100, 0), 2) # 绘制特征提取界限区域
                        cv2.circle(img, (200, 200), 30, (0, 205, 200), 2)
                        img = Image.fromarray(img)
                        img = ImageTk.PhotoImage(image=img)
                        self.video_label.config(image=img)
                        self.video_label.image = img
                        
                    
                # 2. command_face_detection--人脸检测
                elif self.face_detection_active:
                    img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    boxes, _ = self.mtcnn.detect(img, landmarks=False)
                    if boxes is not None:
                        img_array = np.array(img)
                        for box in boxes: 
                            x1, y1, x2, y2 = [int(i) for i in box]  # 绘制人脸检测框
                            width, length = x2 - x1, y2 - y1  # 计算人脸框的宽和高,将其完整的框选出来: 0.1 - 0.3 - 0.1 - 0.1
                            cv2.rectangle(img_array, (x1-int(width*0.1), y1-int(length*0.3)), 
                                          (x2+int(width*0.1), y2+int(length*0.1)), (0, 255, 0), 2)
                        # 播放显示
                        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        img = Image.fromarray(img)
                        img = ImageTk.PhotoImage(image=img)
                        self.video_label.config(image=img)
                        self.video_label.image = img
                        
                # 3. command_face_keypoints_detection--人脸关键点检测
                elif self.face_keypoints_detection_active:
                    faces = self.detector(self.frame, 0)
                    img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    img = np.array(img)
                    
                    if len(faces) != 0:
                        for face in faces:
                            face_shape = self.predictor(self.frame, face)
                            shape = face_utils.shape_to_np(face_shape)
                            for (x, y) in shape:
                                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img = Image.fromarray(img)
                    img = ImageTk.PhotoImage(image=img)
                    self.video_label.config(image=img)
                    self.video_label.image = img
                
                # 4.command_face_aligment--对齐显示
                elif self.face_alignment_active:
                    img_orig = cv2.resize(self.frame,(400,400)) # 保留原图
                    img_rgb = cv2.resize(cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB),(400,400))
                    boxes,_ =self.mtcnn.detect(img_rgb, landmarks=False)
                    face_detection_results = self.face_mesh.process(img_rgb)
                    def draw_rotated_bbox(image, M, center, x1, y1, x2, y2):
                        # 在图像上绘制旋转后的边界框
                        width =  abs(x2 - x1) 
                        height =  abs(y2 - y1)
                        # 计算边界框的四个顶点
                        vertices = np.array([
                            [x1 - width*0.1, y1 - height*0.3],
                            [x2 + width*0.1, y1 - height*0.3],
                            [x2 + width*0.1, y2 + height*0.1],
                            [x1 - width*0.1, y2 + height*0.1]], dtype=np.float32)
                        rotated_vertices = cv2.transform(np.array([vertices]), M)[0]
                        rotated_vertices = rotated_vertices.astype(int)
                        cv2.polylines(image, [rotated_vertices], isClosed=True, color=(0, 255, 0), thickness=2)
                        return image
                    
                    if face_detection_results.multi_face_landmarks and boxes is not None:
                        for face_landmarks,box in zip(face_detection_results.multi_face_landmarks,boxes):
                            # 绘制面部轮廓
                            self.mp_drawing.draw_landmarks(
                                        image=img_orig,
                                        landmark_list=face_landmarks,
                                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                            self.mp_drawing.draw_landmarks(
                                        image=img_orig,
                                        landmark_list=face_landmarks,
                                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                            self.mp_drawing.draw_landmarks(image = img_orig, 
                                         landmark_list=face_landmarks, 
                                         connections = self.mp_face_mesh.FACEMESH_IRISES,
                                         landmark_drawing_spec = None,
                                         connection_drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 128,0), thickness=2, circle_radius=1))
                            # 绘制倾斜四边形:计算M -> 旋转原图 -> 加边界框 -> 旋回原图 
                            h,w,_ = img_orig.shape # 获取图像尺寸ul-251, ur-21, dr-378, dl-149
                            points = [[lm.x*w, lm.y*h] for i,lm in enumerate(face_landmarks.landmark) if i in [54,284,397,172]]
                            points = np.array([points[0],points[2],points[3],points[1]],dtype=np.int32) # 得到最小的四边形
                            angle = math.degrees(math.atan2(points[1][1]-points[0][1],points[1][0]-points[0][0]))

                            x1,y1,x2,y2 = [int(i) for i in box]
                            center = ((x1+x2)//2, (y1+y2)//2)
                            M = cv2.getRotationMatrix2D(center, -angle, 1.0) # 注意角度取反
                            img_rotated = draw_rotated_bbox(img_orig.copy(), M, center, x1, y1, x2, y2)
                        img = Image.fromarray(cv2.cvtColor(img_rotated, cv2.COLOR_RGB2BGR))
                        img = ImageTk.PhotoImage(image=img)
                        self.video_label.config(image=img)
                        self.video_label.image = img

                        if self.toggle_face_alignment_crop_image_active:
                            try:
                                # 进行人脸对齐、裁剪
                                M_rotated = cv2.getRotationMatrix2D(center, angle, 1.0) # 计算旋转后的矩阵
                                img_crop_rotated = cv2.warpAffine(img_rgb, M_rotated, (400,400))
                                box,_ =self.mtcnn.detect(img_crop_rotated, landmarks=False)
                                x1,y1,x2,y2 = [int(i) for i in box[0]]
                                img_crop_save = img_crop_rotated[y1-int(0.3*abs(y2-y1)) : y2 + int(0.1*abs(y2-y1)), x1- int(0.1*abs(x2-x1)) : x2 + int(0.1*abs(x2-x1))] # 裁剪出人脸
                                Image.fromarray(img_crop_save).save('./aligment/face_alignment_crop.jpg') # 保存裁剪后的人脸(400,400)

                                img_crop = cv2.resize(img_crop_save,(180,180))
                                img_crop = Image.fromarray(img_crop)
                                img_crop = ImageTk.PhotoImage(image=img_crop)
                                self.processed_image_label.config(image=img_crop)
                                self.processed_image_label.image = img_crop
                                self.toggle_face_alignment_crop_image_active = not self.toggle_face_alignment_crop_image_active # 关闭裁剪功能
                            except: pass
                    else:
                        img = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
                        img = ImageTk.PhotoImage(image=img)
                        self.video_label.config(image=img)
                        self.video_label.image = img
                        
                # 5.command_face_alignment_crop_image--截取人脸图像
                elif self.toggle_face_alignment_crop_image_active and self.face_alignment_active==False:
                    # 随机截取视频图像
                    img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    Image.fromarray(img).save('./aligment/frame_screenshot.jpg') # 保存截取的图像(400,400)

                    img_show = cv2.resize(img,(180,180))
                    img = Image.fromarray(img_show)
                    img = ImageTk.PhotoImage(image=img)
                    self.processed_image_label.config(image=img)
                    self.processed_image_label.image = img
                    self.toggle_face_alignment_crop_image_active = not self.toggle_face_alignment_crop_image_active # 关闭裁剪功能
                
                # 6. command_face_liveness_detection--活体检测
                elif self.toggle_face_liveness_detection_active:
                    img_rgb = cv2.resize(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), (400,400))
                    face_detection_results = self.face_mesh.process(img_rgb)
                    if face_detection_results.multi_face_landmarks:
                        cv2.circle(img_rgb, (200, 200), 150, (255, 100, 0), 2) # 绘制活体检测区域
                        for face_landmarks in face_detection_results.multi_face_landmarks:
                            h, w, _ = img_rgb.shape  # 瞳孔顺序分别是上-右-下-左
                            DEs = face_landmarks.landmark
                            left_eye = np.array([[DEs[159].x*w, DEs[159].y*h],[DEs[133].x*w, DEs[133].y*h],[DEs[145].x*w, DEs[145].y*h],[DEs[33].x*w, DEs[33].y*h]],dtype=np.int32)
                            right_eye = np.array([[DEs[386].x*w, DEs[386].y*h],[DEs[263].x*w, DEs[263].y*h],[DEs[374].x*w, DEs[374].y*h],[DEs[463].x*w, DEs[463].y*h]],dtype=np.int32)
                            middle_mouth = np.array([[DEs[13].x*w, DEs[13].y*h],[DEs[308].x*w, DEs[308].y*h],[DEs[14].x*w, DEs[14].y*h],[DEs[78].x*w, DEs[78].y*h]],dtype=np.int32)
                            left_eye_ratio, right_eye_ratio, mouth_ratio = self.liveness_compute_ratio(left_eye, right_eye, middle_mouth)
                            cv2.polylines(img_rgb, [left_eye], isClosed=True, color=(0, 255, 0), thickness=2) # 绘制出来便于检测
                            cv2.polylines(img_rgb, [right_eye], isClosed=True, color=(0, 255, 0), thickness=2)
                            cv2.polylines(img_rgb, [middle_mouth], isClosed=True, color=(0, 255, 0), thickness=2)

                            if self.eye_num < 4:
                                print(f'Eye ratio Left is {left_eye_ratio} and Right is {right_eye_ratio}')
                                self.message_label = tk.Label(self.root,text='活体检测:\n实时检测画面中人脸是否为活体\n请眨眨眼三次:'+str(self.eye_num),justify=tk.LEFT,wraplength=160,
                                      fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place(x=25, y=50)
                                flag = False # 判断过程是否连续
                                if (left_eye_ratio < 0.15 and right_eye_ratio < 0.15) : self.eye_count_framenum += 1  
                                elif (left_eye_ratio > 0.2 and right_eye_ratio > 0.2) and self.eye_count_framenum >= 2: self.eye_num += 1;self.eye_count_framenum = 0
                                
                            if  self.eye_num >= 4 and self.mouth_num < 4:
                                if mouth_ratio > 0.5 : self.mouth_count_framenum += 1
                                elif mouth_ratio < 0.2 and self.mouth_count_framenum >= 2: self.mouth_num += 1;self.mouth_count_framenum = 0
                                self.message_label = tk.Label(self.root,text='活体检测:\n实时检测画面中人脸是否为活体\n请张闭口三次:'+str(self.mouth_num),justify=tk.LEFT,wraplength=160,
                                      fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place(x=25, y=50)
                                
                            if self.mouth_num >= 4:
                                tmp_root = tk.Toplevel(self.root)
                                self.center_window(tmp_root,title='活体检测结果')
                                message = tk.Message(tmp_root, text='状态一切正常',width=150).pack()
                                tmp_button = tk.Button(tmp_root, text='关闭', command=lambda:tmp_root.destroy()).pack()
                                tmp_root.wait_window(tmp_root) # 等待用户关闭  
                                self.message_label = tk.Label(self.root,text='               \n'*4,justify=tk.LEFT,wraplength=160,
                                      fg='black',font=('仿宋',14,'bold'),padx=5,pady=10).place(x=25, y=50)
                                self.toggle_face_liveness_detection() # 关闭活体检测功能
                        img = Image.fromarray(img_rgb)
                        img = ImageTk.PhotoImage(image=img)
                        self.video_label.config(image=img)
                        self.video_label.image = img    
                        
                # 7. command_face_recognition--人脸识别
                elif self.face_recognition_active:
                    self.skip_frame_num += 1   # 跳帧自加

                    img_rgb = cv2.resize(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), (400,400))
                    faces_query =self.mtcnn_recognizer(Image.fromarray(img_rgb))
                    boxes,_ = self.mtcnn_recognizer.detect(img_rgb, landmarks=False)
                    if os.path.exists('./mtcnn/facial_feature.csv'):
                        faces_database = pd.read_csv('./mtcnn/facial_feature.csv')
                        faces_database_features = faces_database['特征'].apply(lambda x: [float(i) for i in x.split(',')]) # 获取数据库的人脸特征
                        faces_database_names = faces_database['姓名'] # 获取数据库的人脸姓名

                        try:
                            if faces_query is not None and boxes is not None:
                                for face, box in zip(faces_query, boxes):
                                    x1,y1,x2,y2 = [int(i) for i in box]
                                    cv2.rectangle(img_rgb, (x1,y1), (x2,y2), (0,255,0), 2) # 绘制出人脸框
                                    if self.skip_frame_num % 8 == 0: # 每8帧-跳帧处理
                                        face = self.mtcnn(Image.fromarray(img_rgb[y1:y2,x1:x2])) # 截取人脸区域，单独重新生成face进行识别
                                        face_query_embedding = self.resnet(face.unsqueeze(0)).detach().cpu().numpy()
                                        # 记录每一个face_query与数据库的欧几里得距离
                                        query_database_distance = [np.linalg.norm(face_query_embedding-feature) for feature in faces_database_features]
                                        min_index = np.argmin(query_database_distance)
                                        cv2.rectangle(img_rgb, (x1,y1), (x2,y2), (0,255,0), 2) # 绘制出匹配到的人脸框
                                        img_rgb_plusname = Image.fromarray(img_rgb)
                                        self.face_recognized_name = faces_database_names[min_index]

                                        if query_database_distance[min_index] < 0.8:
                                            ImageDraw.Draw(img_rgb_plusname).text((x1, y1 - 20), faces_database_names[min_index], font=self.font, fill=(0, 255,0)) # 绘制出匹配到的人脸姓名
                                        else:
                                            ImageDraw.Draw(img_rgb_plusname).text((x1, y1 - 20), '未知', font=self.font, fill=(0, 255, 0)) # 绘制出匹配不到人脸姓名
                                    else:
                                        img_rgb_plusname = Image.fromarray(img_rgb)
                                        ImageDraw.Draw(img_rgb_plusname).text((x1, y1-20), self.face_recognized_name, font=self.font, fill=(0, 255,0))
                                img = ImageTk.PhotoImage(image=img_rgb_plusname)
                                self.video_label.config(image=img)
                                self.video_label.image = img   
                        except: pass
                    else:
                        if faces_query is not None and boxes is not None:
                            for face, box in zip(faces_query, boxes):
                                x1,y1,x2,y2 = [int(i) for i in box]
                                cv2.rectangle(img_rgb, (x1,y1), (x2,y2), (0,255,0), 2) # 绘制出人脸框
                                img_rgb_plusname = Image.fromarray(img_rgb)
                                ImageDraw.Draw(img_rgb_plusname).text((x1, y1 - 20), '未知', font=self.font, fill=(0, 255, 0)) # 绘制出匹配不到人脸姓名
                            img = ImageTk.PhotoImage(image=img_rgb_plusname)
                            self.video_label.config(image=img)
                            self.video_label.image = img   
                        else:
                            img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(img)
                            img = ImageTk.PhotoImage(image=img)
                            self.video_label.config(image=img)
                            self.video_label.image = img  

                # # 9. 多种人脸特征显示
                # if self.toggle_show_multi_keypoints_active:
                #     img_rgb = cv2.resize(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), (400,400))
                #     face_detection_results = self.face_mesh.process(img_rgb)
                    
                # 8.常规播放实时视频
                else:
                    img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = ImageTk.PhotoImage(image=img)
                    self.video_label.config(image=img)
                    self.video_label.image = img

                self.after(35)  # 每15ms更新一次视频帧
            else:break
        # 停止视频更新线程
        self.stop_thread = False  

    def center_window(self,toplevel,title=None): 
        # 确保弹出的屏幕居中 
        toplevel.title(title)
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        root_x = self.root.winfo_rootx() + int((root_width - toplevel.winfo_reqwidth()) / 2)
        root_y = self.root.winfo_rooty() + int((root_height - toplevel.winfo_reqheight()) / 2)
        toplevel.geometry("300x50+%d+%d" % (root_x, root_y))
        toplevel.lift()
    
    def clear_show_image(self):
        # 清除显示图像
        self.processed_image_label.config(image='')
    def save_show_image(self):
        # 保存至任意自定义路径,保存显示图像
        try :
            if hasattr(self.processed_image_label, 'image') and self.processed_image_label.image is not None:
                img_save = ImageTk.getimage(self.processed_image_label.image) # 转换成RGB-B格式
                img_save = cv2.resize(cv2.cvtColor(np.array(img_save), cv2.COLOR_RGBA2RGB), (400,400))
                img_save = Image.fromarray(img_save).convert('RGB')
                file_save_path = filedialog.asksaveasfilename(defaultextension='.jpg', filetypes=[('JPG FILE','*.jpg'),('PNG FILE','*.png'),('TIF FILE','*.tif')])
                img_save.save(file_save_path)
        except : pass
    
    def liveness_compute_ratio(self,left_eye, right_eye, mouth):
        # 计算左眼、右眼以及口的上下-左右比率
        def ratio(points_list):
            U_D = math.sqrt((points_list[0][0]-points_list[2][0])**2 + (points_list[0][1]-points_list[2][1])**2)
            L_R = math.sqrt((points_list[1][0]-points_list[3][0])**2 + (points_list[1][1]-points_list[3][1])**2)
            return U_D/L_R
        left_eye_ratio = ratio(left_eye)
        right_eye_ratio = ratio(right_eye)
        mouth_ratio = ratio(mouth)
        return left_eye_ratio, right_eye_ratio, mouth_ratio


"""主程序"""
if __name__ == '__main__':
    root = tk.Tk()
    app = FaceRecognitionAPP(root=root)
    app.create_buttons()
    app.mainloop()