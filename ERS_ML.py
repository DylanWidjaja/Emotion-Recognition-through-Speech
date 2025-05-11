import soundfile # to read audio file
import numpy as np
import pandas as pd
import librosa # to extract speech features
import glob
import os
import pickle # to save model after training
from sklearn.model_selection import train_test_split # for splitting training and testing
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
from sklearn.metrics import accuracy_score # to measure how good we are
#Uncharted territory beyond this line ---------------------------------------------------------------
import tensorflow as tensorflow
from tensorflow import keras 
from tensorflow.keras import layers
from voice import record_to_file
root = "/Users/dwidjaja/Documents/SER"
os.chdir(root)

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result

# all emotions on RAVDESS dataset
int2emotion = {
    "01": "neutral",
    "02": "sadness",
    "03": "anxiety",
    "04": "anger",
    "05": "frustration",
}

#"Emotion (1: \"neutral\", 2: \"sadness\", 3: \"anxiety\", 4: \"anger\", 5: \"frustration\",  6: \"error\"
# we allow only these emotions ( feel free to tune this on your need )
AVAILABLE_EMOTIONS = {
    "neutral",
    "sadness",
    "anxiety",
    "anger",
    "frustration",
}

def load_data(test_size=0.2):
    X, y = [], []
    #for file in glob.glob("data/data/Actor_*/*.wav"):
    for file in glob.glob("labeled_audio/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the emotion label
        #emotion = int(basename.split("-")[2])-1
        emotion = (basename.split('_')[2]).split(".")[0]
        #print(basename.split('_')[1])
        i = 0
        for e in AVAILABLE_EMOTIONS:
            if e == emotion:
                emotion = int(i)
                break
            i+=1
        # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        X.append(features)
        y.append(emotion)
    # split the data to training and testing and return it
    return train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=5)
    # load RAVDESS dataset, 75% training 25% testing
X_train, X_test, y_train, y_test = load_data(test_size=0.25)

# print some details
# number of samples in training data
print("[+] Number of training samples:", X_train.shape)
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape)
# number of features used
# this is a vector of features extracted 
# using extract_features() function
# print("[+] Number of features:", X_train.shape)
# print("hello capy", X_test.shape)
X_train = X_train.reshape(293,1,180)
X_test = X_test.reshape(98,1,180)
# print(X_train.shape)
# print(X_test.shape)
print("\n")
# print(y_train.shape)

# LSTM
model = keras.models.Sequential()
model.add(layers.LSTM(64,input_shape=(1,180),dropout=0.1))
model.add(layers.Dense(5, activation="sigmoid"))
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]
# model.summary()

model.compile(loss=loss, optimizer=optim, metrics=metrics)
model.fit(X_train,y_train, epochs=40, validation_data=(X_test, y_test))

# predict 25% of data to measure how good we are
y_pred = model.predict(X_test)
# print(y_pred)
# print("\n")
#print(y_test)

y_pred_by_max = np.array([])
for pred in y_pred: 
    y_pred_by_max = np.append(y_pred_by_max, np.array(pred.argmax(axis =0)))
#print(y_pred_by_max)
#"Emotion (1: \"neutral\", 2: \"sadness\", 3: \"anxiety\", 4: \"anger\", 5: \"frustration\",  6: \"error\"
emotions = {
    0: "neutral",
    1: "sadness",
    2: "anxiety",
    3: "anger",
    4: "frustration"
}

y_test_text = np.array([])
for emotion_int in y_test:
    y_test_text = np.append(y_test_text, np.array(emotions[int(emotion_int)]))
# print(y_test_text)
y_pred_by_max_text = np.array([])
for emotion_int in y_pred_by_max:
    y_pred_by_max_text = np.append(y_pred_by_max_text, np.array(emotions[int(emotion_int)]))
# print(y_pred_by_max_text)


total = len(y_test)
count = 0
for i in range(total):
    if y_pred_by_max[i] == y_test[i]:
        count+=1
# print("Result: ",count/total*100, "%")


result = pd.DataFrame(y_pred_by_max_text)
result.to_csv("result.csv")

# now we save the model
# make result directory if doesn't exist yet
if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/lstm.model", "wb"))

while True:
    print("Please talk")
    filename = "test1.wav"
    # record the file (start talking)
    record_to_file(filename)
    # extract features and reshape it
    features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
    features = features.reshape(1, 1, 180)
    result = model.predict(features)
    result = emotions[int(np.max(result))]
    # show the result !
    print("result:", result)