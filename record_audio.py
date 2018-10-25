import pyaudio
from recorder import Recorder
import wave
import sys

def play_wav(filename):
	wf = wave.open(filename, 'rb')
	CHUNK = 1024
	p = pyaudio.PyAudio()

	stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
			channels=wf.getnchannels(),
			rate=wf.getframerate(),
			output=True)

	data = wf.readframes(CHUNK)

	while data != '':
	    stream.write(data)
	    data = wf.readframes(CHUNK)

	stream.stop_stream()
	stream.close()

	p.terminate()


res = raw_input('Please press Enter when you are ready to start recording')

rec = Recorder(channels=2)
with rec.open('audio_file_1.wav', 'wb') as recfile:
        recfile.record(duration=5.0)


res = raw_input('Would you like to listen the recording before proceding? y/N')

while res=='y' :
        play_wav('audio_file_1.wav')
        res = raw_input('Would you like to hear the recording again? y/N')

