import numpy as np
from mido import MidiFile

def midi_to_piano_roll(midi_path, fs=100):
    mid = MidiFile(midi_path)
    length = int(mid.length * fs)
    piano_roll = np.zeros((128, length))

    for track in mid.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                piano_roll[msg.note, current_time:] = 1
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                piano_roll[msg.note, current_time:] = 0
    return piano_roll
