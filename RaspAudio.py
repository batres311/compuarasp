import RPi.GPIO as GPIO
import yaml
import pyaudio #Libreria que ayuda para obtener el audio y darle formato
import wave  #Permite leer y escribir archivos wav
import scipy.io.wavfile as waves #libreria importante para los datos del audio

with open("Variables.yaml", "r") as f:
    yaml_content = yaml.full_load(f)

FRAME_RATE = yaml_content["Frame_rate"]
CHANNELS = yaml_content["Channels"]
FRAMESPERBUFFER= yaml_content["FramesPerBuffer"]
Empezar = yaml_content["BotonGrabar"]
Detener = yaml_content["BotonDetener"]

class GetAudiosandSetup():
    def __init__(self,audio,archivo):
        
        self.audio=audio
        self.archivo=archivo


    def setup():
        GPIO.setwarnings(False) 
        GPIO.setmode(GPIO.BOARD)       # Numbers GPIOs by physical location
        # Set Green Led Pin mode to output
        GPIO.setup(Detener, GPIO.IN, pull_up_down=GPIO.PUD_UP)      # Set Red Led Pin mode to output
        GPIO.setup(Empezar, GPIO.IN, pull_up_down=GPIO.PUD_UP) 
    
    def loop(audio,archivo):
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