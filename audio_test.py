from recorder import Recorder

res = raw_input('Please press Enter when you are ready to start recording')

rec = Recorder(channels=2)
with rec.open('audio_file_1.wav', 'wb') as recfile:
    recfile.record(duration=5.0)
