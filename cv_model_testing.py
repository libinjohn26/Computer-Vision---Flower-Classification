import cv2
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score

model = tf.keras.models.load_model("cv_cnn_model.model")

pickle_in = open("X_test.pickle","rb")
x = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y = pickle.load(pickle_in)

y_binary = to_categorical(y)

x = x/255.0

#prediction = model.predict(x)
prediction = model.predict_classes(x,batch_size = 10)
print(prediction)
print(y)

cm = confusion_matrix(y,prediction)
print(accuracy_score(y, prediction)) # normalize=False))

plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()