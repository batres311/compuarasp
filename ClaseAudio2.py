import numpy as np
import librosa
import os
import yaml

class CargaeImagenAudio():
    def __init__(self,clip,path_export1,path_export2,res,NombreImag,fig,archivo):
        
        self.clip=clip
        self.path_export1=path_export1
        self.path_export2=path_export2
        self.res=res  
        self.NombreImag=NombreImag 
        self.fig=fig
        self.archivo=archivo
    def LoadAudio_Turn2Decibels(clip):
        y, sr = librosa.load(clip, sr=44100) 
        D = librosa.stft(y) 
        # STFT of y 
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) 
        #, ref=np.max

        return y,S_db,sr
    
    def guardarimagen(path_export1,path_export2,res,NombreImag,fig,archivo):
    
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