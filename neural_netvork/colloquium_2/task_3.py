from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def create_LeNet_model():
    model = Sequential()

    model.add(Conv2D(6, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


LeNet_model = create_LeNet_model()
print(LeNet_model.summary())
