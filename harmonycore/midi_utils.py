from note_seq.protobuf import music_pb2
from note_seq import sequences_lib, chord_inference
import note_seq
import pretty_midi

def save_note_sequence_as_midi(note_sequence: music_pb2.NoteSequence, midi_path: str):
    note_seq.sequence_proto_to_midi_file(note_sequence, midi_path)


def load_midi_to_note_sequence(midi_path: str) -> music_pb2.NoteSequence:
    return note_seq.midi_file_to_sequence_proto(midi_path)


def note_sequence_to_chords(note_sequence: music_pb2.NoteSequence):
    # Infère les accords à partir de la séquence de notes (note_seq)
    chord_sequence = chord_inference.infer_chords_for_sequence(note_sequence)
    chords = [e.text for e in chord_sequence.text_annotations
              if e.annotation_type == music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL]
    return chords


def plot_piano_roll(note_sequence: music_pb2.NoteSequence):
    note_seq.plot_sequence(note_sequence)
