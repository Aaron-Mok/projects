# from music21 import converter, note, chord
# from collections import defaultdict

# def parse_sheet(filepath):
#     score = converter.parse(filepath)
#     parts = score.parts

#     def group_notes(part):
#         groups = defaultdict(list)
#         for n in part.flat.notes:
#             # Round offset to avoid float mismatch (e.g., 0.000001 vs 0.0)
#             key = round(n.offset, 3)
#             if isinstance(n, note.Note):
#                 groups[key].append(n.nameWithOctave)
#             elif isinstance(n, chord.Chord):
#                 groups[key].extend(p.nameWithOctave for p in n.pitches) #n.pitches have two notes, G4 and E5
#         return [groups[o] for o in sorted(groups.keys())]

#     treble_notes = group_notes(parts[0])
#     bass_notes = group_notes(parts[1]) if len(parts) > 1 else []

#     print("🎼 Treble clef notes:", treble_notes)
#     print("🎹 Bass clef notes:", bass_notes)

#     return treble_notes, bass_notes

# treble_list, bass_list = parse_sheet("assets/scores/BOOK-Richard-Clayderman-Music-of-Love_p1.mxl")
# print("🎼 Treble clef notes:", treble_list)

# from music21 import converter, environment, note, chord

# environment.set('musicxmlPath', r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe')  # adjust path if needed

# def parse_sheet(filepath):
#     score = converter.parse(filepath)
#     parts = score.parts

#     treble_notes = []
#     bass_notes = []

#     for element in parts[0].flat.notes:  # Right hand
#         if isinstance(element, note.Note):
#             treble_notes.append(element.nameWithOctave)
#         elif isinstance(element, chord.Chord):
#             treble_notes.extend(n.nameWithOctave for n in element.notes)

#     for element in parts[1].flat.notes:  # Left hand
#         if isinstance(element, note.Note):
#             bass_notes.append(element.nameWithOctave)
#         elif isinstance(element, chord.Chord):
#             bass_notes.extend(n.nameWithOctave for n in element.notes)

#     print("🎼 Treble clef notes:", treble_notes)
#     print("🎹 Bass clef notes:", bass_notes)

#     return treble_notes, bass_notes


# treble_list, bass_list = parse_sheet("assets/scores/BOOK-Richard-Clayderman-Music-of-Love_p1.mxl")

from music21 import converter, note, chord, stream
from collections import defaultdict

def parse_sheet(filepath):
    score = converter.parse(filepath)
    
    # Combine all parts into one flat stream (treble + bass)
    combined = score.parts.stream()  # Keeps timing info intact
    all_notes = combined.flat.notes  # Flattened list of notes from both staves

    groups = defaultdict(list)
    for n in all_notes:
        key = round(n.offset, 3)
        if isinstance(n, note.Note):
            groups[key].append(n.nameWithOctave)
        elif isinstance(n, chord.Chord):
            groups[key].extend(p.nameWithOctave for p in n.pitches)

    # Return as a time-ordered list of note groups
    return [groups[o] for o in sorted(groups.keys())]


notes = parse_sheet("assets/scores/BOOK-Richard-Clayderman-Music-of-Love_p1.mxl")
print(notes)
