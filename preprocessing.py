import pretty_midi
import numpy as np
import os
import functools

def concatNotes(a, b):
    return a + (b.notes if hasattr(b, 'notes') else [])

def processMidi(midi_file):
    """
    Convert MIDI file to a Piano Roll array.
    :param midi_file: Path to the MIDI file.
    :return: Piano roll of shape (num_pitches, time_steps).
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    def noteToVector(note):
        """
        Convert a note to a one-hot vector representation.
        :param note: Note to convert.
        :return: One-hot vector representation of the note.
        """
        return (
            note.pitch,
            midi_data.time_to_tick(note.start),
            midi_data.time_to_tick(note.end) - midi_data.time_to_tick(note.start),
        )

    allNotes = list(
        sorted(
            map(
                noteToVector,
                functools.reduce(concatNotes, midi_data.instruments, []),
            ),
            key=lambda x: x[1],
        )
    )

    if not allNotes:
        return np.array([])  # Handle empty notes list

    final_note = np.array(
        [
            (
                0,
                midi_data.time_to_tick(allNotes[-1][1] + allNotes[-1][2]),
                0,
            )
        ]
    )

    return np.concatenate(
        (
            np.array(allNotes),
            final_note
        ),
        axis=0,
    )

midi_directory = "./midis"
fileList = []
for root, dirs, files in os.walk(midi_directory):
    for file in files:
        if file.endswith(".mid"):
            fileList.append(os.path.join(root, file))

if not os.path.exists("output"):
    os.mkdir("output")

for file in fileList:
    try:
        notes = processMidi(file)
        if notes.size == 0:
            print("No notes found in", file)
            continue
        # Normalize the file path for cross-platform compatibility
        normalized_file_path = os.path.normpath(file)
        output_filename = os.path.join("output", normalized_file_path.replace(midi_directory + os.sep, "").replace(os.sep, "-").replace(".mid", "") + ".npy")
        np.save(output_filename, notes)
        print("Processed", file)
    except Exception as e:
        print("Error with", file, ":", str(e))
