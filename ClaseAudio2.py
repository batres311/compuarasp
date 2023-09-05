import numpy as np
import librosa
import os
import yaml
import pandas as pd
import pyaudio #Libreria que ayuda para obtener el audio y darle formato
import wave  #Permite leer y escribir archivos wav
import scipy.io.wavfile as waves #libreria importante para los datos del audio

import shutil #libreria para mover archivos a diferentes carpetas
from pydub import AudioSegment
import RPi.GPIO as GPIO
import time

with open("Variables.yaml", "r") as f:
    yaml_content = yaml.full_load(f)

FRAME_RATE = yaml_content["Frame_rate"]
CHANNELS = yaml_content["Channels"]
FRAMESPERBUFFER= yaml_content["FramesPerBuffer"]
FRAME_SIZE = yaml_content["Frame_size"]
RAW= yaml_content["Raw"]
CLEAN = yaml_content["Clean"]
PRODUCTO=yaml_content["Producto"]
Empezar = yaml_content["BotonGrabar"]
Detener = yaml_content["BotonDetener"]
GRABAR=yaml_content["Grabar"]
LISTO=yaml_content["Listo"]
ESPERA=yaml_content["Espera"]
DURACION=yaml_content["duracion"]

class CargaeImagenAudio():
    def __init__(self,clip,y,rate,threshold,path_export1,path_export2,res,
                 NombreImag,fig,archivo,year,month,day,path_espacio,path_estado,path_programa,
                 extract,linea,estacion,audios):
        self.clip=clip
        self.y=y
        self.rate=rate
        self.threshold=threshold
        self.path_export1=path_export1
        self.path_export2=path_export2
        self.res=res  
        self.NombreImag=NombreImag 
        self.fig=fig
        self.archivo=archivo
        self.year=year
        self.month=month
        self.day=day
        self.path_espacio=path_espacio
        self.path_estado=path_estado
        self.path_programa=path_programa
        self.extract=extract
        self.linea=linea
        self.estacion=estacion
        self.audios=audios

    def setup():
        GPIO.setwarnings(False) 
        GPIO.setmode(GPIO.BOARD)       # Numbers GPIOs by physical location
        # Set Green Led Pin mode to output
        GPIO.setup(Detener, GPIO.IN, pull_up_down=GPIO.PUD_UP)      # Set Red Led Pin mode to output
        GPIO.setup(Empezar, GPIO.IN, pull_up_down=GPIO.PUD_UP) 
        GPIO.setup(GRABAR, GPIO.OUT)
        GPIO.setup(LISTO, GPIO.OUT)
        GPIO.setup(ESPERA, GPIO.OUT)

    def GuardaAudio(year, month,day,linea,estacion,audios,path_programa):
        if not os.path.exists(PRODUCTO): 
            os.makedirs(PRODUCTO, exist_ok=True) 
        path_base=os.path.join(path_programa,PRODUCTO)
        path_linea=os.path.join(path_base,linea) 
        path_estacion=os.path.join(path_linea,estacion)
        path_trabajo=os.path.join(path_estacion,audios)
        path_year = os.path.join(path_trabajo, year)
        path_month = os.path.join(path_year, month)
        path_day = os.path.join(path_month, day)
        path_raw=os.path.join(path_day,RAW)
        path_clean=os.path.join(path_day,CLEAN)
        
        if os.path.isdir(path_base) == True:
            if os.path.isdir(path_linea) == False:
                os.mkdir(path_linea)
            if os.path.isdir(path_linea) == True:
                if os.path.isdir(path_estacion) == False:
                    os.mkdir(path_estacion)
                if os.path.isdir(path_estacion) == True:       # si la carpeta no existe, entonces crea la carpeta
                        if os.path.isdir(path_trabajo) == False:
                            os.mkdir(path_trabajo)
                        if os.path.isdir(path_trabajo)==True:
                            if os.path.isdir(path_year)==False:
                                os.mkdir(path_year)
                            if os.path.isdir(path_year) == True:       # si la carpeta no existe, entonces crea la carpeta
                                if os.path.isdir(path_month) == False:
                                    os.mkdir(path_month)
                                if os.path.isdir(path_month)==True:
                                    if os.path.isdir(path_day)==False:
                                        os.mkdir(path_day)
                                    if os.path.isdir(path_day)==True:
                                        if os.path.isdir(path_raw)==False:
                                            os.mkdir(path_raw)
                                    if os.path.isdir(path_day)==True:
                                        if os.path.isdir(path_clean)==False:
                                            os.mkdir(path_clean)

                                    return path_raw,path_clean,path_day
                                
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
    
    def loop(audio,archivo,year,month,day,linea,estacion,audios,path_programa):
        while True: 
            if GPIO.input(Empezar)==0:                                                                                                                                                                                                                                                                                  
                stream=audio.open(format=pyaudio.paInt16,channels=CHANNELS,
                                    rate=FRAME_RATE,input=True, #rate es la frecuencia de muestreo 44.1KHz
                                    frames_per_buffer=FRAMESPERBUFFER)
                GPIO.output(LISTO,0)
                time.sleep(0.2) 
                GPIO.output(GRABAR,1)
                            
                print("Grabando ...") #Mensaje de que se inicio a grabar
                frames=[] #Aqui guardamos la grabacion
                #for i in range(0,int(44100/1024*duracion)):
                for i in range(0,int(FRAME_RATE/FRAMESPERBUFFER*DURACION)):
                    data=stream.read(FRAMESPERBUFFER)
                    frames.append(data)

                    
                stream.stop_stream()    #Detener grabacion
                stream.close()          #Cerramos stream
                audio.terminate()
                time.sleep(0.5) 
                GPIO.output(GRABAR,0)
                time.sleep(0.2) 
                
                #print("La grabacion ha terminado ") #Mensaje de fin de grabación
                path_raw,path_clean,path_day=CargaeImagenAudio.GuardaAudio(year,month,day,
                                                                            linea,estacion,audios,path_programa)

                waveFile=wave.open(archivo,'wb') #Creamos nuestro archivo
                waveFile.setnchannels(CHANNELS) #Se designan los canales
                waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                waveFile.setframerate(FRAME_RATE) #Pasamos la frecuencia de muestreo
                waveFile.writeframes(b''.join(frames))
                waveFile.close() #Cerramos el archivo
                return path_raw, path_clean, path_day
    
    def loop2():
        GPIO.add_event_detect(Empezar,GPIO.FALLING)
        GPIO.add_event_detect(Detener,GPIO.FALLING)
        while True:
            GPIO.output(ESPERA,1)
            time.sleep(0.25)
            GPIO.output(ESPERA,0)
            time.sleep(0.25)
            if GPIO.event_detected(Empezar):
                GPIO.remove_event_detect(Empezar)
                GPIO.remove_event_detect(Detener)
                res="ok"
                print("La grabacion se ha clasificado como ok")
                return res
                break
            if GPIO.event_detected(Detener):
                GPIO.remove_event_detect(Detener)
                GPIO.remove_event_detect(Empezar)
                res="nok"
                print("La grabacion se ha clasificado como nok")
                return res
                break

    def AcomodoPathRAW(path_espacio,path_estado,path_programa,archivo):
        os.chdir(path_espacio)
        if not os.path.exists(path_estado): 
            os.makedirs(path_estado, exist_ok=True) 
        path_oknok=os.path.join(path_espacio,path_estado)
        os.chdir(path_programa)
        path_final=os.path.join(path_oknok,archivo)
        shutil.move(archivo, path_final)

    def AcomodoPathClean(path_espacio,path_estado,extract,archivo):
        os.chdir(path_espacio)
        if not os.path.exists(path_estado): 
            os.makedirs(path_estado, exist_ok=True) 
        path_oknok=os.path.join(path_espacio,path_estado)
        path_final=os.path.join(path_oknok,archivo)
        extract.export(path_final, format="wav")
        os.chdir(path_oknok)

    def LoadAudio_Turn2Decibels(clip):
        y, sr = librosa.load(clip, sr=44100) 
        D = librosa.stft(y) 
        # STFT of y 
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) 
        #, ref=np.max
        return y,S_db,sr
    
    def guardarimagen(path_day,path_actual,path_export1,path_export2,res,NombreImag,fig,archivo):
    
        os.chdir(path_day)
        audio_filename=archivo
        image_filename_to_save = str(audio_filename).replace(".wav", "_", 1) + NombreImag+".png" 
        if not os.path.exists(path_export1): 
            os.makedirs(path_export1, exist_ok=True) 
        if not os.path.exists(path_export2): 
            os.makedirs(path_export2, exist_ok=True)
        if res=='ok'or res=='OK' or res=='Ok':
            image_filename_to_save2 ="Bosch"+image_filename_to_save 
            fig.savefig(os.path.join(path_export1,image_filename_to_save2))
        else:
            image_filename_to_save2 ="BlackDecker"+image_filename_to_save 
            fig.savefig(os.path.join(path_export2,image_filename_to_save2))
        os.chdir(path_actual)
