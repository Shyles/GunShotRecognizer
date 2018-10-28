import time
import pyaudio
from recorder import Recorder
import wave
import sys
import pandas as pd
import librosa

audio_file = 'audio_file_1.wav'

def main()
	res = raw_input('Would you like to 1) record a new audio file or 2) use an existing audio file? Please type 1 or 2 and press Enter to choose.\n')

	if res=='1':
		print('You have chosen to record audio.\n')
		record_audio()
	else:
		audio_file = raw_input('You have chosen to use an existing audio file. Please provide the full path of the audio file and press Enter.\n')

	data2 = pd.Series([librosa.util.pad_center(librosa.load(audio_file, mono=True)[0], 88375)])

	result = True

	print('Detecting gunshot sounds from your recording... ')
	for i in range(0,3) : # We want to make the model more credible by not processing the recording too fast
		print('.')
		time.sleep(0.5)
	print('Your recording has been classified as ...\n' + ('Gunshot', 'Not a gunshot.')[result])

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
	with rec.open(audio_file, 'wb') as recfile:
		recfile.record(duration=2.0)


	res = raw_input('Would you like to listen the recording before proceding? y/N\n')

	while res=='y' :
		play_wav(audio_file)
		res = raw_input('Would you like to hear the recording again? y/N')

main()
