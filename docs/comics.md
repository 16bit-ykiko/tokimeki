# Manga and novels, voiced (planned)

There is no reader here. You keep reading in your usual app and send what is on screen to a small HTTP service on this machine; it answers with audio.

1. **Capture.** iOS allows no app to float over other apps, but AssistiveTouch (or Back Tap, or the Action Button) can run a Shortcut: *Take Screenshot* → *Get Contents of URL* (POST to the service) → play the audio it returns. No app to build or install. Android can have a real floating button; a desktop a hotkey. The phone reaches the PC on the same Wi-Fi, or from outside through a VPN such as Tailscale.
2. **Read the page.** A vision model returns a script: the panels in reading order and, per bubble, the speaker, the text, the delivery (shy, teasing, whispering …), plus narration and sound effects. Speakers are matched to the series cast by name, and by CCIP where a face is in the panel.
3. **Voice it.** Each line in its character's voice model, narration in a neutral voice, joined with short pauses and returned as one audio file. Later: a motion-comic video that pans across the panels in time with the lines, rendered and subtitled by the MAD renderer.

Novels take the same path without the vision step: the text goes to a language model for speakers and delivery. Whether pages need a local rating pass before they reach a cloud model is still open; most vision models accept typical manga pages.
