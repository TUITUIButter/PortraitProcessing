import numpy
from .base_module import BaseModule
import cv2


class Layout(BaseModule):
    def __init__(self, path='modules/graph_opt.pb'):
        # 加载模型
        self.net = cv2.dnn.readNetFromTensorflow(path)

    def cal_score(self, img) -> float:
        self.get_pose(img)
        return 0

    def opt_img(self, img) -> numpy.ndarray:
        res = [1, 2, 3]
        return numpy.array(res)
        pass

    """
    获取19个关键点
    """
    def get_pose(self, img):
        # 获取图像的高度和宽度
        (h, w) = img.shape[:2]

        # 创建一个 blob 对象
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (1000, 1000)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # 将 blob 对象传递给模型
        self.net.setInput(blob)

        # 运行模型
        output = self.net.forward()

        BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                      "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
        POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                      ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                      ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                      ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = output[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (w * point[0]) / output.shape[3]
            y = (h * point[1]) / output.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > 0.2 else None)

        print(points)

        # 划线
        for pair in POSE_PAIRS:
            part_from = pair[0]
            part_to = pair[1]
            assert (part_from in BODY_PARTS)
            assert (part_to in BODY_PARTS)

            id_from = BODY_PARTS[part_from]
            id_to = BODY_PARTS[part_to]

            if points[id_from] and points[id_to]:
                cv2.line(img, points[id_from], points[id_to], (0, 255, 0), 3)
                cv2.ellipse(img, points[id_from], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(img, points[id_to], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
