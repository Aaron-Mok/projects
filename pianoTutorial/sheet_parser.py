from music21 import converter, note, chord
from collections import defaultdict

def parse_sheet(filepath):
    score = converter.parse(filepath)

    # Use part index to determine treble (0) or bass (1)
    part_labels = {0: "treble", 1: "bass"}
    grouped = defaultdict(lambda: {"treble": [], "bass": [], "measure": None})

    for i, part in enumerate(score.parts):  # e.g., treble, bass
        label = part_labels.get(i, f"part{i}")
        for n in part.flat.notes:
            key = round(n.offset, 3)
            measure = n.getContextByClass("Measure")
            mnum = measure.number if measure else 1

            if isinstance(n, note.Note):
                grouped[key][label].append(n.nameWithOctave)
            elif isinstance(n, chord.Chord):
                grouped[key][label].extend(p.nameWithOctave for p in n.pitches)

            grouped[key]["measure"] = mnum

    # Return list of (treble_notes, bass_notes, measure)
    return [
        (entry["treble"], entry["bass"], entry["measure"])
        for offset, entry in sorted(grouped.items())
    ]

# from music21 import converter, note, chord, stream
# from collections import defaultdict

# from music21 import converter, note, chord, stream
# from collections import defaultdict

# def parse_sheet(filepath):
#     score = converter.parse(filepath)

#     # Combine all parts into one flat stream
#     combined = score.parts.stream()
#     all_notes = combined.flat.notes

#     # Group notes by time offset and record measure number
#     groups = defaultdict(lambda: {"notes": [], "measure": None})

#     for n in all_notes:
#         key = round(n.offset, 3)
#         measure = n.getContextByClass("Measure")
#         measure_number = measure.number if measure else 1

#         if isinstance(n, note.Note):
#             groups[key]["notes"].append(n.nameWithOctave)
#         elif isinstance(n, chord.Chord):
#             groups[key]["notes"].extend(p.nameWithOctave for p in n.pitches)

#         groups[key]["measure"] = measure_number

#     # Sort and return list of (note_group, measure_number)
#     return [(groups[o]["notes"], groups[o]["measure"]) for o in sorted(groups.keys())]


# def parse_sheet(filepath):
#     score = converter.parse(filepath)
    
#     # Combine all parts into one flat stream (treble + bass)
#     combined = score.parts.stream()  # Keeps timing info intact
#     all_notes = combined.flat.notes  # Flattened list of notes from both staves

#     groups = defaultdict(list)
#     for n in all_notes:
#         key = round(n.offset, 3)
#         if isinstance(n, note.Note):
#             groups[key].append(n.nameWithOctave)
#         elif isinstance(n, chord.Chord):
#             groups[key].extend(p.nameWithOctave for p in n.pitches)

#     # Return as a time-ordered list of note groups
#     return [groups[o] for o in sorted(groups.keys())]

# THis is another parser version that seperates trble and bass notes.
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

#     return treble_notes, bass_notes