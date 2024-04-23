import pretty_midi
import numpy as np
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def npy_to_midi(npy_path, output_midi_path, default_tempo=120):
    """Converts an NPY file containing note information back into a MIDI file."""
    try:
        note_data = np.load(npy_path)
        if note_data.shape[0] < 2:
            logging.info("Not enough notes to generate a MIDI file for " + npy_path)
            return

        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
        ticks_per_beat = midi.resolution
        time_per_tick = (60.0 / default_tempo) / ticks_per_beat

        for pitch, start_tick, duration_tick in note_data:
            start_time = start_tick * time_per_tick
            end_time = (start_tick + duration_tick) * time_per_tick
            note = pretty_midi.Note(velocity=100, pitch=int(pitch), start=start_time, end=end_time)
            piano.notes.append(note)

        midi.instruments.append(piano)
        midi.write(output_midi_path)
        logging.info(f"MIDI file generated at {output_midi_path}")
    except Exception as e:
        logging.error(f"Failed to convert {npy_path} to MIDI: {str(e)}")

if __name__ == "__main__":
    input_npy_path = 'path_to_npy_file.npy'
    output_midi_path = 'path_to_output_midi_file.mid'
    npy_to_midi(input_npy_path, output_midi_path)
