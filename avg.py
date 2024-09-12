import os
import numpy as np
import cv2

root_path = 'E:/All_application_resouces/dataset/bed/output2'

frequency = [1, 64]
images_cnt = 3
height = 256
width = 256
i = 0
for path1 in os.listdir(root_path):
    img = np.zeros((height, width, 3), np.uint8)
    for filename in os.listdir(os.path.join(root_path, path1)):
        path = os.path.join(root_path, path1, filename)
        img1 = cv2.imread(path)
        img += img1

    img = (img/images_cnt).astype(np.uint8)
    cv2.imwrite('E:/All_application_resouces/dataset/bed/output2'+str(frequency[i])+'.jpg', img)
    i += 1

