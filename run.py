#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score

#Reading two CSV files
test = pd.read_csv("sign_mnist_test.csv")
train = pd.read_csv("sign_mnist_train.csv")

#storing labels
labels = train['label'].values

#bar chart visualization
plt.figure(figsize = (18,8))
sns.countplot(x=labels)

#removing 'label' column from train dataframe
train.drop('label', axis=1, inplace=True)

#reshaping images
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

#transforming original categorical labels into binary representation
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)

#splitting the data into 30% testing and 70% training
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=101)

#scaling and reshaping
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#building CNN model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(24, activation='softmax'))  # 24 classes (excluding J and Z for sign language)
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

#training the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128)

#saving the model
model.save("sign_mnist_cnn_model.h5")
print("Model Saved")

#plotting accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])
plt.show()

#testing with test data
test_labels = test['label']
test.drop('label', axis=1, inplace=True)
test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
test_images = test_images / 255.0

#predicting on test set
y_pred = model.predict(test_images)
print("Test Accuracy:", accuracy_score(test_labels, y_pred.round()))

#function to map prediction to a letter
def getLetter(result):
    classLabels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
                   7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O',
                   14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
                   20: 'V', 21: 'W', 22: 'X', 23: 'Y'}
    try:
        return classLabels[result]
    except:
        return "Error"

#setting up webcam for gesture detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]
    box_width, box_height = 300, 300  # Width and height of the ROI box
    x_center = frame_width // 2
    y_center = frame_height // 2

    # Calculate top-left and bottom-right coordinates for the ROI box
    top_left_x = x_center - box_width // 2
    top_left_y = y_center - box_height // 2
    bottom_right_x = x_center + box_width // 2
    bottom_right_y = y_center + box_height // 2

    # Extract the ROI
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] # region of interest (ROI) for gesture
    cv2.imshow('ROI', roi)

    #preprocessing the image from the webcam
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)  # resize to 28x28
    roi = roi / 255.0  # normalize
    roi = roi.reshape(1, 28, 28, 1)  # reshape for model input

    #predicting gesture
    result = model.predict(roi)
    confidence = np.max(result)

    #if confidence is above threshold, display predicted letter, else "No Gesture"
    if confidence > 0.8:  # confidence threshold
        predicted_letter = getLetter(np.argmax(result))
    else:
        predicted_letter = "No Gesture"

    #displaying prediction on the frame
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 5)
    cv2.putText(copy, predicted_letter, (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('Frame', copy)

    #exit the loop on 'Enter' key
    if cv2.waitKey(1) == 13:  # 13 is the Enter key
        break

cap.release()
cv2.destroyAllWindows()
