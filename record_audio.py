import time
import pyaudio
from recorder import Recorder
import wave
import sys
import pandas as pd
import librosa
from torch.autograd import Variable
import torch
import time
import os
import datetime

default_audio_file = 'recording.wav'
weights = torch.load('weights.pt')

def main():
	res = raw_input('Would you like to 1) record a new audio file or 2) use an existing audio file or 3) use your microphone to detect gunshots for next 5 minutes? Please type 1,2 or 3 and press Enter to choose.\n')

	if res=='1':
		print('You have chosen to record audio.\n')
		record_audio()
		detect_gunshot(default_audio_file)
	elif res=='2':
		audio_file = raw_input('You have chosen to use an existing audio file. Please provide the full path of the audio file and press Enter.\n')
		detect_gunshot(audio_file)
	else:
		print('You have chosen to detect gunshots with your microphone for 5 minutes.\n Starting now... \n')
		timestr = time.strftime("%Y%m%d-%H%M%S")
		if os.path.exists('recordings') == False:
			os.mkdir('recordings')
		for i in range(1, 150): # Record 2 second audios for 150 times equals 5 minutes
			audio_file = 'recordings/' + str(timestr) + '_recording_' + str(i) + '.wav'
			rec = Recorder(channels=2)
			with rec.open(audio_file, 'wb') as recfile:
				recfile.record(duration=2.0)
			print('Gunshot has been detected at ' + str(datetime.datetime.now()))


def load_wav(filename):
     data, sample_rate = librosa.load('audio_falie_1.wav',res_type='kaiser_fast')
     feat = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
     return Variable(torch.from_numpy(np.array([feat]))).float()

def detect_gunshot(filename):
	print('Detecting gunshot sounds from your recording... ')
	for i in range(0,3) : # We want to make the model more credible by not processing the recording too fast
		print('.')
		time.sleep(0.5)
	print('Your recording has been classified as ...\n' + ('Gunshot', 'Not a gunshot.')[classify_gunshot(filename)])

def classify_gunshot(filename):
	"""
	:param x: full or relative path of the wav-file
	:return: True if audio is a gunshot with higher than 97% probability, otherwise false
	"""
	wav_file = load_wav(filename)
	predi = torch.nn.functional.softmax(model(wav_file,weights),dim=1)
	return predi.data.cpu().numpy()[0][1] > 0.97

def model(x, weights):
    """
    :param x: Input vector
    :param weights: ANN weights
    :return: Output vector of ANN
    """
    w1, b1, w2, b2 = weights
    return torch.mm(torch.sigmoid(torch.mm(x,w1)+b1),w2)+b2

def play_wav(filename):
	wf = wave.open(filename, 'rb')
	CHUNK = 1024
	p = pyaudio.PyAudio()

	stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
			channels=wf.getnchannels(), rate=wf.getframerate(), output=True)

	data = wf.readframes(CHUNK)

	while data != '':
	    stream.write(data)
	    data = wf.readframes(CHUNK)

	stream.stop_stream()
	stream.close()

	p.terminate()

def record_audio():
	res = raw_input('Please press Enter when you are ready to start recording\n')

	rec = Recorder(channels=2)
	with rec.open(default_audio_file, 'wb') as recfile:
		recfile.record(duration=2.0)


	res = raw_input('Would you like to listen the recording before proceding? y/N\n')

	while res=='y' :
		play_wav(audio_file)
		res = raw_input('Would you like to hear the recording again? y/N')

main()
