import time

class NoteTrainer:
    def __init__(self, notes, flash_callback):
        self.notes = notes
        self.flash_callback = flash_callback
        self.current_index = 0
        self.flash_note()

    def flash_note(self):
        if self.current_index < len(self.notes):
            note = self.notes[self.current_index]
            print(f"🎯 Expecting: {note}")
            self.flash_callback(note)
        else:
            print("🎉 All notes played!")

    def on_note_detected(self, detected):
        if not detected:
            return

        target_note = self.notes[self.current_index]
        if detected[0] == target_note:
            print(f"✅ Correct: {detected[0]}")
            self.current_index += 1
            self.flash_note()
        else:
            print(f"❌ Heard {detected[0]}, but expecting {target_note}")
