from music21 import converter, environment, note, chord

environment.set('musicxmlPath', r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe')  # set to MuseScore executable

score = converter.parse("BOOK-Richard-Clayderman-Music-of-Love_p1.mxl")
parts = score.parts

treble_notes = []
bass_notes = []

sheet_notes = []
for element in parts[0].flat.notes:  # Treble (right hand)
    if isinstance(element, note.Note):
        treble_notes.append(element.nameWithOctave)
    elif isinstance(element, chord.Chord):
        for n in element.notes:
            treble_notes.append(n.nameWithOctave)

for element in parts[1].flat.notes:  # Bass (left hand)
    if isinstance(element, note.Note):
        bass_notes.append(element.nameWithOctave)
    elif isinstance(element, chord.Chord):
        for n in element.notes:
            bass_notes.append(n.nameWithOctave)

print("🎼 Treble clef notes:", treble_notes)
print("🎹 Bass clef notes:", bass_notes)