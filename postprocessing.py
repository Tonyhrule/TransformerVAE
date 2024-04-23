import pretty_midi
import numpy as np
import os

def npy_to_midi(npy_path, output_midi_path, default_tempo=120):
    """
    Converts an NPY file containing note information back into a MIDI file.
    This function assumes npy data format: [pitch, start_tick, duration_tick].
    """
    # Load note data from the npy file
    note_data = np.load(npy_path)
    
    # Check if there are enough notes to create a MIDI file
    if note_data.shape[0] < 2:
        print("Not enough notes to estimate tempo and generate a MIDI file.")
        return

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    # Create an instrument instance
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    # Calculate time per tick based on the default tempo and resolution
    ticks_per_beat = midi.resolution
    time_per_tick = (60.0 / default_tempo) / ticks_per_beat

    # Add notes to the piano instrument
    for note_info in note_data:
        pitch, start_tick, duration_tick = map(int, note_info)
        start_time = start_tick * time_per_tick
        end_time = (start_tick + duration_tick) * time_per_tick
        note = pretty_midi.Note(
            velocity=100, pitch=pitch, start=start_time, end=end_time)
        piano.notes.append(note)

    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(piano)
    # Save to MIDI file
    midi.write(output_midi_path)
    print(f"Generated MIDI file saved to {output_midi_path}")
