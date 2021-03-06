from flask import Flask, render_template, request, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
import cv2


# model = Sequential()
# model.add(preprocessing.RandomTranslation(0.25, 0.25))
# model.add(preprocessing.RandomRotation(0.5))
# model.add(preprocessing.Normalization())
# model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='Same'))
# model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='Same'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='Same'))
# model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='Same'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

# model.add(Flatten())
# model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.2))
# model.add(Dense(4, activation='softmax'))

    
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.load_weights('static/model.h5')

model = load_model('./static/model.h5')


COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (112,112))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 112,112,3)
    prediction = model.predict(img_arr)

    a = round(prediction[0,0], 2)
    b = round(prediction[0,1], 2)
    c = round(prediction[0,2], 2)
    preds = np.array([a, b, c])
    COUNT += 1
    return render_template('prediction.html', data=preds)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    app.jinja_env.globals.update(max=max)

