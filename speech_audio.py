import speech_recognition as sr
import pyaudio
import wave
import time
import threading
import os
import winsound  # For Windows sound playback

# Define the threshold for noise detection
NOISE_THRESHOLD = 500  # Adjust this value as needed

def play_alert_sound():
    winsound.Beep(1000, 500)  # Frequency of 1000 Hz for 500 milliseconds

def detect_background_noise(source):
    r = sr.Recognizer()
    audio = r.listen(source)
    if audio:
        try:
            volume = max(audio.get_raw_data())
            if volume > NOISE_THRESHOLD:
                play_alert_sound()
        except Exception as e:
            print(f"Error in noise detection: {e}")

def read_audio(stream, filename):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    seconds = 10  # Number of seconds to record at once
    frames = []  # Initialize array to store frames

    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
   
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    # Stop and close the stream
    stream.stop_stream()
    stream.close()

def convert(i):
    if i >= 0:
        sound = 'record' + str(i) + '.wav'
        r = sr.Recognizer()
       
        with sr.AudioFile(sound) as source:
            r.adjust_for_ambient_noise(source)
            print("Converting Audio To Text and saving to file..... ")
            detect_background_noise(source)  # Check for background noise
            audio = r.listen(source)
        try:
            value = r.recognize_google(audio)  # API call to Google for speech recognition
            os.remove(sound)
            if str is bytes:
                result = u"{}".format(value).encode("utf-8")
            else:
                result = "{}".format(value)
 
            with open("test.txt", "a") as f:
                f.write(result)
                f.write(" ")
               
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Google Speech Recognition request error: {e}")
        except KeyboardInterrupt:
            pass

p = pyaudio.PyAudio()  # Create an interface to PortAudio
chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100

def save_audios(i):
    stream = p.open(format=sample_format, channels=channels, rate=fs,
                    frames_per_buffer=chunk, input=True)
    filename = 'record' + str(i) + '.wav'
    read_audio(stream, filename)

flag = False
for i in range(30 // 10):  # Number of total seconds to record / Number of seconds per recording
    t1 = threading.Thread(target=save_audios, args=[i])
    x = i - 1
    t2 = threading.Thread(target=convert, args=[x])  # send one earlier than being recorded
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    if i == 2:
        flag = True
if flag:
    convert(i)
    p.terminate()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

file = open("test.txt")  # Student speech file
data = file.read()
file.close()
stop_words = set(stopwords.words('english'))  
word_tokens = word_tokenize(data)  # Tokenizing sentence
filtered_sentence = [w for w in word_tokens if not w in stop_words]  
filtered_sentence = []
 
for w in word_tokens:   # Removing stop words
    if w not in stop_words:
        filtered_sentence.append(w)

# Creating a final file
with open('final.txt', 'w') as f:
    for ele in filtered_sentence:
        f.write(ele + ' ')

# Checking whether proctor needs to be alerted or not
file = open("paper.txt")  # Question file
data = file.read()
file.close()
stop_words = set(stopwords.words('english'))  
word_tokens = word_tokenize(data)  # Tokenizing sentence
filtered_questions = [w for w in word_tokens if not w in stop_words]  
filtered_questions = []
 
for w in word_tokens:   # Removing stop words
    if w not in stop_words:
        filtered_questions.append(w)
       
def common_member(a, b):    
    a_set = set(a)
    b_set = set(b)
     
    # Check length  
    if len(a_set.intersection(b_set)) > 0:
        return a_set.intersection(b_set)  
    else:
        return []

comm = common_member(filtered_questions, filtered_sentence)
print('Number of common elements:', len(comm))
print(comm)