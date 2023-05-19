import cv2
import mediapipe as mp

import time  # 计算fps值

# 两个初始化
mpPose = mp.solutions.pose
pose = mpPose.Pose()
# 初始化画图工具
mpDraw = mp.solutions.drawing_utils

# path = "./imgs/middle-tilt.jpg"  # 倾斜的图片 angle: 25.91°
# path = './imgs/test.png'  # angle: 5.57°
# path = "./imgs/02dark.png"  # angle: 5.38°
# path = "./imgs/cloudy.jpg"  # angle: 5.38°
# path = "./imgs/good-001.JPG"  # 找不到LHip这个点
# path = "./imgs/good-002.JPG"  # 背影无法定位
# path = "./imgs/good-010.JPEG"  # angle: 10.78°
# path = "imgs/bad-001.JPG"  # angle: 3.59°
path = "imgs/03.jpg"  # angle: 3.59°

# 调用摄像头，在同级目录下新建Videos文件夹，然后在里面放一些MP4文件，方便读取
img = cv2.imread(path)  # 倾斜的图片 angle: 25.91°
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray_img)

# 转换为RGB格式，因为Pose类智能处理RGB格式，读取的图像格式是BGR格式
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 处理一下图像
results = pose.process(imgRGB)
# print(results.pose_landmarks)
# 检测到人体的话：
if results.pose_landmarks:
    # 使用mpDraw来刻画人体关键点并连接起来
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    # 如果我们想对33个关键点中的某一个进行特殊操作，需要先遍历33个关键点
    for id, lm in enumerate(results.pose_landmarks.landmark):
        # 打印出来的关键点坐标都是百分比的形式，我们需要获取一下视频的宽和高
        h, w, c = img.shape
        print(id, lm)
        # 将x乘视频的宽，y乘视频的高转换成坐标形式
        cx, cy = int(lm.x * w), int(lm.y * h)
        # 使用cv2的circle函数将关键点特殊处理
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

# cv2.namedWindow('Original', cv2.WINDOW_KEEPRATIO)    # 窗口大小可以改变
cv2.imshow("Original", img)
cv2.waitKey(0)
