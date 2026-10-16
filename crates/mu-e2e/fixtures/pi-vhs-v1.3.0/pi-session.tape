Output "crates/mu-e2e/fixtures/pi-vhs-v1.3.0/pi-session.gif"
Output "crates/mu-e2e/fixtures/pi-vhs-v1.3.0/pi-session.ascii"
Set Shell bash
Set Width 1000
Set Height 700
Set Framerate 15
Set TypingSpeed 15ms
Hide
Type "cd $PI_VHS_PROJECT_DIR"
Enter
Sleep 300ms
Type "$PI_ORACLE_BIN --offline --no-extensions --no-skills"
Enter
Sleep 3s
Show
Sleep 500ms
Type "/"
Sleep 500ms
Type "model"
Sleep 500ms
Escape
Sleep 500ms
Ctrl+C
Type "composer oracle prompt"
Sleep 500ms
Ctrl+C
Type "/quit"
Enter
Sleep 1s
