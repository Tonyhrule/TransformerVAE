import pretty_midi
import numpy as np
import os
import functools
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def concatNotes(a, b):
    """Concatenate notes from two instruments."""
    return a + (b.notes if hasattr(b, 'notes') else [])

def processMidi(midi_file):
    """Convert MIDI file to a numpy array representing a piano roll."""
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    def noteToVector(note):
        return (note.pitch, midi_data.time_to_tick(note.start), midi_data.time_to_tick(note.end) - midi_data.time_to_tick(note.start))

    allNotes = list(sorted(map(noteToVector, functools.reduce(concatNotes, midi_data.instruments, [])), key=lambda x: x[1]))
    if not allNotes:
        return np.array([])  # Handle empty notes list

    final_note = np.array([(0, midi_data.time_to_tick(allNotes[-1][1] + allNotes[-1][2]), 0)])
    return np.concatenate((np.array(allNotes), final_note), axis=0)

def main():
    midi_directory = "./midis"
    output_directory = "./output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    fileList = [os.path.join(root, file) for root, _, files in os.walk(midi_directory) for file in files if file.endswith(".mid")]
    for file in fileList:
        try:
            notes = processMidi(file)
            if notes.size == 0:
                logging.info(f"No notes found in {file}")
                continue
            normalized_file_path = os.path.normpath(file)
            output_filename = os.path.join(output_directory, normalized_file_path.replace(midi_directory + os.sep, "").replace(os.sep, "-").replace(".mid", "") + ".npy")
            np.save(output_filename, notes)
            logging.info(f"Processed {file}")
        except Exception as e:
            logging.error(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    main()
