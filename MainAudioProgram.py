import pyaudio #Libreria que ayuda para obtener el audio y darle formato
#import wave  #Permite leer y escribir archivos wav
import scipy.io.wavfile as waves #libreria importante para los datos del audio
import yaml
import os
import shutil #libreria para mover archivos a diferentes carpetas
from datetime import datetime
import ClaseAudio2
import ClaseFeatures
import wave  #Permite leer y escribir archivos wav
#import winsound #Permite acceder a la maquinaria básica de reproducción de sonidos proporcionada por la plataformas Windows.
import scipy.io.wavfile as waves #libreria importante para los datos del audio
import yaml
import os
import shutil #libreria para mover archivos a diferentes carpetas
from datetime import datetime
import ClaseAudio2
import ClaseFeatures
from pydub import AudioSegment
import librosa
from ctypes import *
from contextlib import contextmanager
import pyaudio #Libreria que ayuda para obtener el audio y darle formato
import RPi.GPIO as GPIO


with open("Variables.yaml", "r") as f:
    yaml_content = yaml.full_load(f)

Bosch_path_export = yaml_content["BuenoMalo"]["Bueno"]
BlackDecker_path_export = yaml_content["BuenoMalo"]["Malo"]
Audiobueno_path_export = yaml_content["AUDIOS"]["BUENO"]
Audiomalo_path_export = yaml_content["AUDIOS"]["MALO"]
STARTSEC= yaml_content["StartSec"]
ENDSEC= yaml_content["EndSec"]
NOMBREGRABACION=yaml_content["NomGrabacion"]
LINEA=yaml_content["Linea"]
ESTACION=yaml_content["Estacion"]
AUDIOS=yaml_content["Audios"]
Empezar = yaml_content["BotonGrabar"]
Detener = yaml_content["BotonDetener"]
GRABAR=yaml_content["Grabar"]
LISTO=yaml_content["Listo"]
ESPERA=yaml_content["Espera"]

FechaHoraAUDIO=datetime.now()
year = str(FechaHoraAUDIO.year)
month = str(FechaHoraAUDIO.month)
day = str(FechaHoraAUDIO.day)
if FechaHoraAUDIO.month <= 9:
    month = "0{}".format(FechaHoraAUDIO.month)
FechaHoraAUDIO=FechaHoraAUDIO.replace(microsecond=0)
FechaHoraAUDIOFormat=FechaHoraAUDIO.strftime("%Y_%m_%d_%H_%M_%S")
archivo=NOMBREGRABACION+"_"+FechaHoraAUDIOFormat +".wav"

if __name__ == '__main__':
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

    ClaseAudio2.CargaeImagenAudio.setup()
    GPIO.output(LISTO,1) 
    path_actual=os.getcwd()    
    print("Listo para grabar presiona el boton de grabar")
    path_raw,path_clean, path_day=ClaseAudio2.CargaeImagenAudio.loop(audio,archivo,year,month,day,LINEA,
                                                                     ESTACION,AUDIOS,path_actual)	
    print("La grabacion ha terminado ") #Mensaje de fin de grabación
    #winsound.PlaySound(archivo,winsound.SND_FILENAME)
    
    GPIO.output(ESPERA,1)
    print("Oprime el boton G para indicar que la grabacion es ok o D para indicar que nok: ")


    res=ClaseAudio2.CargaeImagenAudio.loop2()
    signal,S_db1,sample_rate=ClaseAudio2.CargaeImagenAudio.LoadAudio_Turn2Decibels(archivo)
    base= archivo  

    if res=='ok' or res=='OK' or res=='Ok':
        ClaseAudio2.CargaeImagenAudio.AcomodoPathRAW(path_raw,Audiobueno_path_export,path_actual,archivo)
    else:
        ClaseAudio2.CargaeImagenAudio.AcomodoPathRAW(path_raw,Audiomalo_path_export,path_actual,archivo)

    mask = ClaseAudio2.CargaeImagenAudio.envelope(signal,sample_rate, 0.003)#Bosch=0.004,0.0025
    waves.write(filename="clean"+"Grab"+".wav", rate=sample_rate, data=signal[mask])
    filee="clean"+"Grab"+".wav"
    signal1, rate1 = librosa.load("clean"+"Grab"+".wav", sr=44100)
    mask2 = ClaseAudio2.CargaeImagenAudio.envelope2(signal1, rate1, 0.016)#Bosch=0.0095,0.0097
    waves.write(filename="New"+"clean"+"Grab"+".wav", rate=rate1, data=signal1[mask2])
    filee2="New"+"clean"+"Grab"+".wav"

    if len(mask2)>= 44100: #Multiplicar frame rate por el endsec
        sound = AudioSegment.from_file(file="New"+"clean"+"Grab"+".wav",format="wav")
        AudioSegment.converter="ffmpeg.exe"
        AudioSegment.ffmpeg="ffmpeg.exe"
        AudioSegment.ffprobe="ffprobe.exe"
        StartTime=STARTSEC*1000
        EndTime=ENDSEC *1000
        extract=sound[StartTime:EndTime]
        
        if res=='ok' or res=='OK' or res=='Ok':
            ClaseAudio2.CargaeImagenAudio.AcomodoPathClean(path_clean,Audiobueno_path_export,extract,base)
        else:
            ClaseAudio2.CargaeImagenAudio.AcomodoPathClean(path_clean,Audiomalo_path_export,extract,base)
        y,S_db,sr=ClaseAudio2.CargaeImagenAudio.LoadAudio_Turn2Decibels(base)#"newfile"+str(i)+".wav"
        os.chdir(path_actual)
        os.remove("clean"+"Grab"+".wav")  
        os.remove("New"+"clean"+"Grab"+".wav") 
        ClaseFeatures.Features.waveform(y,sr,res,archivo,path_day,path_actual)
        ClaseFeatures.Features.amplitudeenvelope(y,sr,res,archivo,path_day,path_actual)
        t=ClaseFeatures.Features.RootMeanSquaredError(y,sr,res,archivo,path_day,path_actual)
        ClaseFeatures.Features.ZeroCrossingRate(y,sr,res,t,archivo,path_day,path_actual)
        ClaseFeatures.Features.FreqAmp(y,sr,res,archivo,path_day,path_actual)
        ClaseFeatures.Features.Spectrogram(S_db,sr,res,archivo,path_day,path_actual)
        ClaseFeatures.Features.GreySpectrogram(S_db,sr,res,archivo,path_day,path_actual)
        ClaseFeatures.Features.MelSpectrogram(y,sr,res,archivo,path_day,path_actual)
        ClaseFeatures.Features.Chromagram(y,sr,res,archivo,path_day,path_actual)
        mfccs=ClaseFeatures.Features.MFCCs(y,sr,res,archivo,path_day,path_actual)
        ClaseFeatures.Features.DeltaMFCCs(mfccs,sr,res,archivo,path_day,path_actual)
        ClaseFeatures.Features.Delta2MFCCs(mfccs,sr,res,archivo,path_day,path_actual)
        ClaseFeatures.Features.BandEnergyRatio(y,sr,res,archivo,path_day,path_actual)
        ClaseFeatures.Features.SpectralCentroid(y,sr,t,res,archivo,path_day,path_actual)
        ClaseFeatures.Features.Bandwidht(y,sr,t,res,archivo,path_day,path_actual)
        ClaseFeatures.Features.SpectralContrast(y,sr,res,archivo,path_day,path_actual)
        S=ClaseFeatures.Features.SpectralRollOff(y,S_db,sr,res,archivo,path_day,path_actual)
        ClaseFeatures.Features.PolyFeatures(S,sr,res,archivo,path_day,path_actual)
        ClaseFeatures.Features.Tonnetz(y,sr,res,archivo,path_day,path_actual)
        GPIO.output(ESPERA,0)
    else:
        print("Grabe un audio de mas de 1 segundo y vuelva a usar el programa")
