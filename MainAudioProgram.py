import pyaudio #Libreria que ayuda para obtener el audio y darle formato
import wave  #Permite leer y escribir archivos wav
import winsound #Permite acceder a la maquinaria básica de reproducción de sonidos proporcionada por la plataformas Windows.
import scipy.io.wavfile as waves #libreria importante para los datos del audio
import yaml
import os
import shutil #libreria para mover archivos a diferentes carpetas
from datetime import datetime
import keyboard as kb
import ClaseAudio2
import ClaseFeatures
import pandas as pd
import numpy as np
from pydub import AudioSegment
import librosa

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

NOMBREGRABACION=yaml_content["NomGrabacion"]
i=0

duracion=5 #Periodo de grabacion de 5 segundos
FechaHoraAUDIO=datetime.now()
FechaHoraAUDIO=FechaHoraAUDIO.replace(microsecond=0)
FechaHoraAUDIOFormat=FechaHoraAUDIO.strftime("%Y_%m_%d_%H_%M_%S")
archivo=NOMBREGRABACION+"_"+FechaHoraAUDIOFormat +".wav"

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

def loop():
    while True:
       
        if kb.is_pressed('g'):
            audio=pyaudio.PyAudio() #Iniciamos pyaudio
            #Abrimos corriente o flujo
            stream=audio.open(format=pyaudio.paInt16,channels=CHANNELS,
                                rate=FRAME_RATE,input=True, #rate es la frecuencia de muestreo 44.1KHz
                                frames_per_buffer=FRAME_SIZE)
                                
            print("Grabando ...") #Mensaje de que se inicio a grabar
            print("Presiona p para parar carnal") #Mensaje de que se inicio a grabar
            frames=[] #Aqui guardamos la grabacion

            while True:
                data=stream.read(FRAME_SIZE)
                frames.append(data)
                
                if kb.is_pressed('p'):
                    print('se presionó [p]arar!')
                    stream.stop_stream()    #Detener grabacion
                    stream.close()          #Cerramos stream
                    audio.terminate()

                    waveFile=wave.open(archivo,'wb') #Creamos nuestro archivo
                    waveFile.setnchannels(CHANNELS) #Se designan los canales
                    waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                    waveFile.setframerate(FRAME_RATE) #Pasamos la frecuencia de muestreo
                    waveFile.writeframes(b''.join(frames))
                    waveFile.close() #Cerramos el archivo
                    break
            break
    
print("Listo para grabar presiona g")
loop()	
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


mask = envelope(signal,sample_rate, 0.003)#Bosch=0.004,0.0025
waves.write(filename="clean"+"Grab"+str(i)+".wav", rate=sample_rate, data=signal[mask])
filee="clean"+"Grab"+str(i)+".wav"
signal1, rate1 = librosa.load("clean"+"Grab"+str(i)+".wav", sr=44100)
mask2 = envelope2(signal1, rate1, 0.016)#Bosch=0.0095,0.0097
waves.write(filename="New"+"clean"+"Grab"+str(i)+".wav", rate=rate1, data=signal1[mask2])
filee2="New"+"clean"+"Grab"+str(i)+".wav"
    
#signal2, rate2 = librosa.load("NewcleanGrab1.wav")
sound = AudioSegment.from_file(file="New"+"clean"+"Grab"+str(i)+".wav",format="wav")
AudioSegment.converter="ffmpeg.exe"
AudioSegment.ffmpeg="ffmpeg.exe"
AudioSegment.ffprobe="ffprobe.exe"

startseg=0
endsec=1

StartTime=startseg*1000
EndTime=endsec*1000

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

ClaseFeatures.Features.amplitudeenvelope(y,res,archivo)

t=ClaseFeatures.Features.RootMeanSquaredError(y,res,archivo)

ClaseFeatures.Features.ZeroCrossingRate(y,res,t,archivo)

ClaseFeatures.Features.FreqAmp(y,sr,res,archivo)

ClaseFeatures.Features.Spectrogram(S_db,res,archivo)

ClaseFeatures.Features.GreySpectrogram(S_db,res,archivo)

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

ClaseFeatures.Features.PolyFeatures(S,res,archivo)

ClaseFeatures.Features.Tonnetz(y,sr,res,archivo)
