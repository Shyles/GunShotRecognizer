# GunShotRecognizer

## Contents

This project contains a model for detecting gun shots in urban environments. Data sets are not provided, but they can be found in the links given at the end of the README.

## Usage



You might be interested to load the dataset we used from [Kaggle](https://www.kaggle.com/pavansanagapati/urban-sound-classification). You can then train your own model by tinkering with the `gunshotclassifier.py`. Just for running the program to detect gunshots, you can run the `executable.py`.

If you want jsut to use our model and detect gunshots using it, you'll have three options after you start the program.

### Record your own audio
You can record your own 2 second long audio file by entering 1 after the first prompt. You will be asked if you want to listen to the audio before getting your prediction on whether the recording contains a gunshot.

### Use an existing audio file
You may have an existing audio file you would like to process. By choosing option 2, you will be asked for relative (to the current directory) or full path of your file. After you've passed the path, the program will tell you its prediction.

### Use microphone continuously to detect gunshot
Last option will use your microphone for 5 minutes to detect gunshot. The internal logic in the detection is that every 2 seconds, the program creates an wav-file to a new directory called recordings. These files will then immediately be processed and there will be an output for each detection of gunshot. These audio files are never deleted, so you may want to save up on disk space and remove the directory manually every now and then.

