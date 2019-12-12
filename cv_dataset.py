import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from scipy.io import loadmat
import random
import pickle

dataFile = "C:\\Users\\john\\Desktop\\Libin Docs\\Study Material\\Computer Vision\\Student Project\\jpg"
imageLabels = loadmat('imagelabels.mat')
classValue = imageLabels['labels']
labels = classValue[0]

path = os.path.join(dataFile)
image_size = 100
i = 0
training_data = []
for img in tqdm(os.listdir(path)):  # iterate over each image
    try:
        class_value = labels[i]
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_UNCHANGED)# ,cv2.IMREAD_GRAYSCALE)  # convert to array
        new_array = cv2.resize(img_array, (image_size, image_size))  # resize to normalize data size
        training_data.append([new_array, class_value])  # add this to our training_data
        i = i + 1
        #plt.imshow(new_array, cmap='gray')  # graph it
        #plt.show()  # display!
    except Exception as e:  # in the interest in keeping the output clean...
        pass
    #except OSError as e:
    #    print("OSErrroBad img most likely", e, os.path.join(path,img))
    #except Exception as e:
    #    print("general exception", e, os.path.join(path,img))


random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, image_size, image_size, 1)
y = np.array(y)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
