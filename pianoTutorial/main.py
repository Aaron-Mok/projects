if __name__ == "__main__":
    import time
    import threading
    from vision.keyboard_ui import KeyboardUI
    from sheet_parser import parse_sheet
    from audio.mic_listener import MicListener
    from render_utils import render_measure  # or directly use if defined above
    from music21 import converter

    ui = KeyboardUI()
    test_notes = parse_sheet("assets/scores/BOOK-Richard-Clayderman-Music-of-Love_p1.mxl")
    score = converter.parse("assets/scores/BOOK-Richard-Clayderman-Music-of-Love_p1.mxl")
    current_index = [0]  # Mutable container to allow update from callback

    # ✅ Pre-render each unique measure
    unique_measures = set(m for (_, _, m) in test_notes)
    measure_images = {m: render_measure(score, m) for m in unique_measures}

    def handle_detected_notes(detected):
        if not ui.locked or current_index[0] >= len(test_notes):
            return

        treble, bass, measure_number = test_notes[current_index[0]]
        expected_notes = treble + bass

        if all(n in detected for n in expected_notes):
            print(f"🎯 Matched: {expected_notes}")
            current_index[0] += 1
            if current_index[0] < len(test_notes):
                treble_next, bass_next, next_measure = test_notes[current_index[0]]
                ui.flash_note({"treble": treble_next, "bass": bass_next})
                ui.show_measure_image(measure_images[next_measure])
            else:
                print("🎉 All notes completed!")
                ui.flash_note(None)

    # Start UI and Listener in separate threads
    listener = MicListener(callback=handle_detected_notes)
    listener_thread = threading.Thread(target=listener.run)
    ui_thread = threading.Thread(target=ui.show)

    ui_thread.start()
    listener_thread.start()

    # Wait until the keyboard is locked before starting
    while not ui.locked:
        time.sleep(0.1)

    if test_notes:
        treble, bass, measure = test_notes[0]
        print("🚀 Starting trainer...")
        ui.flash_note({"treble": treble, "bass": bass})
        ui.show_measure_image(measure_images[measure])

    ui_thread.join()
    print("🛑 UI exited. Stopping listener...")
    listener.stop()
    listener_thread.join()
    print("✅ Done.")