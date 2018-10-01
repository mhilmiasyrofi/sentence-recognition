from gtts import gTTS
import os
tts = gTTS(text='Selamat Malam', lang='id')
tts.save("audio/output.mp3")
os.system("mpg321 audio/output.mp3")