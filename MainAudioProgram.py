import pyaudio #Libreria que ayuda para obtener el audio y darle formato
import wave  #Permite leer y escribir archivos wav
import scipy.io.wavfile as waves #libreria importante para los datos del audio
import yaml
import os
import shutil #libreria para mover archivos a diferentes carpetas
from datetime import datetime
import ClaseAudio2
import ClaseFeatures
import pandas as pd
import numpy as np
from pydub import AudioSegment
import librosa
import RPi.GPIO as GPIO
from ctypes import *
from contextlib import contextmanager

with open("Variables.yaml", "r") as f:
    yaml_content = yaml.full_load(f)



Bosch_path_export = yaml_content["BuenoMalo"]["Bueno"]
BlackDecker_path_export = yaml_content["BuenoMalo"]["Malo"]

CarpetaRoBosch=yaml_content["CarpetasROW"]["BOSCH"]
CarpetaRoBlackDecker=yaml_content["CarpetasROW"]["BLACKDECKER"]

FRAME_SIZE = yaml_content["Frame_size"]
HOP_LENGTH = yaml_content["Hop_lenght"]
FRAME_RATE = yaml_content["Frame_rate"]
CHANNELS = yaml_content["Channels"]
NUMBER_MELS = yaml_content["Number_Mels"]
N_FTT = yaml_content["N_fft"]
N_MFCC = yaml_content["Number_MFCCs"]
HOP_SIZE= yaml_content["Hop_size"]
FRAMESPERBUFFER= yaml_content["FramesPerBuffer"]

THRESHOLD1= yaml_content["Threshold1"]
THRESHOLD2= yaml_content["Threshold2"]
STARTSEC= yaml_content["StartSec"]
ENDSEC= yaml_content["EndSec"]
Empezar = yaml_content["BotonGrabar"]
Detener = yaml_content["BotonDetener"]

NOMBREGRABACION=yaml_content["NomGrabacion"]
i=0

duracion=5 #Periodo de grabacion de 5 segundos
FechaHoraAUDIO=datetime.now()
FechaHoraAUDIO=FechaHoraAUDIO.replace(microsecond=0)
FechaHoraAUDIOFormat=FechaHoraAUDIO.strftime("%Y_%m_%d_%H_%M_%S")
archivo=NOMBREGRABACION+"_"+FechaHoraAUDIOFormat +".wav"

def setup():
	GPIO.setwarnings(False) 
	GPIO.setmode(GPIO.BOARD)       # Numbers GPIOs by physical location
	   # Set Green Led Pin mode to output
	GPIO.setup(Detener, GPIO.IN, pull_up_down=GPIO.PUD_UP)      # Set Red Led Pin mode to output
	GPIO.setup(Empezar, GPIO.IN, pull_up_down=GPIO.PUD_UP) 
        
def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/100), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return np.array(mask)

def envelope2(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/100), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean < threshold:
            mask.append(True)
        else:
            mask.append(False)
    return np.array(mask)

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

with noalsaerr():
    audio=pyaudio.PyAudio() #Iniciamos pyaudio
#Abrimos corriente o flujo

def loop(audio):
    while True: 
        if GPIO.input(Empezar)==0:                                                                                                                                                                                                                                                                                  
            stream=audio.open(format=pyaudio.paInt16,channels=CHANNELS,
                                rate=FRAME_RATE,input=True, #rate es la frecuencia de muestreo 44.1KHz
                                frames_per_buffer=FRAMESPERBUFFER)
                        
            print("Grabando ...") #Mensaje de que se inicio a grabar
            frames=[] #Aqui guardamos la grabacion
            #for i in range(0,int(44100/1024*duracion)):
            while True:
                data=stream.read(FRAMESPERBUFFER)
                frames.append(data)

                if GPIO.input(Detener)==0: 
                    stream.stop_stream()    #Detener grabacion
                    stream.close()          #Cerramos stream
                    audio.terminate()
                    #print("La grabacion ha terminado ") #Mensaje de fin de grabación

                    waveFile=wave.open(archivo,'wb') #Creamos nuestro archivo
                    waveFile.setnchannels(CHANNELS) #Se designan los canales
                    waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                    waveFile.setframerate(FRAME_RATE) #Pasamos la frecuencia de muestreo
                    waveFile.writeframes(b''.join(frames))
                    waveFile.close() #Cerramos el archivo
                    break
            break

setup()   
print("Listo para grabar presiona g")
loop(audio)	
print("La grabacion ha terminado ") #Mensaje de fin de grabación
#clip=(r'C:\Users\BHC4SLP\Documents\Python Projects\Proyecto2-GraficaAudio\PruebaAudio1.wav')
#winsound.PlaySound(archivo,winsound.SND_FILENAME)
res=input("Ingresa ok si es buena grabacion y nok si es mala: ")
signal,S_db1,sample_rate=ClaseAudio2.CargaeImagenAudio.LoadAudio_Turn2Decibels(archivo)
if not os.path.exists(Bosch_path_export): 
    os.makedirs(Bosch_path_export, exist_ok=True) 
if not os.path.exists(BlackDecker_path_export): 
    os.makedirs(BlackDecker_path_export, exist_ok=True)
base= archivo   
if res=='ok' or res=='OK' or res=='Ok':
    shutil.move(archivo, CarpetaRoBosch+"/"+archivo)
    os.rename(CarpetaRoBosch+"/"+archivo, CarpetaRoBosch+"/"+CarpetaRoBosch+archivo)
else:
     shutil.move(archivo, CarpetaRoBlackDecker+"/"+archivo)
     os.rename(CarpetaRoBlackDecker+"/"+archivo, CarpetaRoBlackDecker+"/"+CarpetaRoBlackDecker+archivo)   


mask = envelope(signal,sample_rate, THRESHOLD1)#Bosch=0.004,0.0025
waves.write(filename="clean"+"Grab"+str(i)+".wav", rate=sample_rate, data=signal[mask])
filee="clean"+"Grab"+str(i)+".wav"
signal1, rate1 = librosa.load("clean"+"Grab"+str(i)+".wav", sr=FRAME_RATE)
mask2 = envelope2(signal1, rate1, THRESHOLD2)#Bosch=0.0095,0.0097
waves.write(filename="New"+"clean"+"Grab"+str(i)+".wav", rate=rate1, data=signal1[mask2])
filee2="New"+"clean"+"Grab"+str(i)+".wav"
    
#signal2, rate2 = librosa.load("NewcleanGrab1.wav")
sound = AudioSegment.from_file(file="New"+"clean"+"Grab"+str(i)+".wav",format="wav")
AudioSegment.converter="ffmpeg.exe"
AudioSegment.ffmpeg="ffmpeg.exe"
AudioSegment.ffprobe="ffprobe.exe"

StartTime=STARTSEC*1000
EndTime=ENDSEC*1000

extract=sound[StartTime:EndTime]
extract.export(base, format="wav")
y,S_db,sr=ClaseAudio2.CargaeImagenAudio.LoadAudio_Turn2Decibels(base)#"newfile"+str(i)+".wav"
if res=='ok' or res=='OK' or res=='Ok':
    shutil.move(base, Bosch_path_export+"/"+base)
    os.rename(Bosch_path_export+"/"+base, Bosch_path_export+"/"+"BoschClean"+base)
else:
     shutil.move(base, BlackDecker_path_export+"/"+base)
     os.rename(BlackDecker_path_export+"/"+base, BlackDecker_path_export+"/"+"BlackDeckerClean"+base) 
os.remove("clean"+"Grab"+str(i)+".wav")  
os.remove("New"+"clean"+"Grab"+str(i)+".wav") 

ClaseFeatures.Features.waveform(y,sr,res,archivo)

ClaseFeatures.Features.amplitudeenvelope(y,sr,res,archivo)

t=ClaseFeatures.Features.RootMeanSquaredError(y,sr,res,archivo)

ClaseFeatures.Features.ZeroCrossingRate(y,res,t,archivo)

ClaseFeatures.Features.FreqAmp(y,sr,res,archivo)

ClaseFeatures.Features.Spectrogram(S_db,sr,res,archivo)

ClaseFeatures.Features.GreySpectrogram(S_db,sr,res,archivo)

ClaseFeatures.Features.MelSpectrogram(y,sr,res,archivo)

ClaseFeatures.Features.Chromagram(y,sr,res,archivo)

mfccs=ClaseFeatures.Features.MFCCs(y,sr,res,archivo)

ClaseFeatures.Features.DeltaMFCCs(mfccs,sr,res,archivo)

ClaseFeatures.Features.Delta2MFCCs(mfccs,sr,res,archivo)

ClaseFeatures.Features.BandEnergyRatio(y,sr,res,archivo)

ClaseFeatures.Features.SpectralCentroid(y,sr,t,res,archivo)

ClaseFeatures.Features.Bandwidht(y,sr,t,res,archivo)

ClaseFeatures.Features.SpectralContrast(y,sr,res,archivo)

S=ClaseFeatures.Features.SpectralRollOff(y,S_db,sr,res,archivo)

ClaseFeatures.Features.PolyFeatures(S,sr,res,archivo)

ClaseFeatures.Features.Tonnetz(y,sr,res,archivo)
