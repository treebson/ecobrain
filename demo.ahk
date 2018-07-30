msgbox, Initiating EcoBrain Demo

time = 100

run terminal.bat
sleep %time%
sendinput python ecobrain.py
sleep %time%
sendinput {Enter}
sleep %time%
sendinput no
sleep %time%
sendinput {Enter}
sleep %time%
sendinput {LWin down}
sleep %time%
sendinput {Left}
sleep %time%
sendinput {Up}
sleep %time%
sendinput {LWin up}
sleep %time%
sleep %time%

run terminal.bat
sleep %time%
sendinput python image.py
sleep %time%
sendinput {Enter}
sleep %time%
sendinput {LWin down}
sleep %time%
sendinput {Right}
sleep %time%
sendinput {Up}
sleep %time%
sendinput {LWin up}
sleep %time%
sleep %time%

run terminal.bat
sleep %time%
sendinput python sound.py
sleep %time%
sendinput {Enter}
sleep %time%
sendinput no
sleep %time%
sendinput {Enter}
sleep %time%
sendinput {LWin down}
sleep %time%
sendinput {Left}
sleep %time%
sendinput {Down}
sleep %time%
sendinput {LWin up}
sleep %time%
sleep %time%

run terminal.bat
sleep %time%
sendinput python series.py
sleep %time%
sendinput {Enter}
sleep %time%
sendinput {LWin down}
sleep %time%
sendinput {Right}
sleep %time%
sendinput {Down}
sleep %time%
sendinput {LWin up}
sleep %time%
