import librosa
import ClaseAudio2
import matplotlib.pyplot as plt
import yaml
import numpy as np
import math

with open("Variables.yaml", "r") as f:
    yaml_content = yaml.full_load(f)

WAVEFORM_path_export1 =yaml_content["WAVEFORM"]["Carpetaok"]
WAVEFORM_path_export2 = yaml_content["WAVEFORM"]["Carpetanok"]
AMPLITUDEENV_path_export1=yaml_content["AMPLITUDEENV"]["Carpetaok"]
AMPLITUDEENV_path_export2=yaml_content["AMPLITUDEENV"]["Carpetanok"]
RMSE_path_export1=yaml_content["RMSE"]["Carpetaok"]
RMSE_path_export2=yaml_content["RMSE"]["Carpetanok"]
ZCR_path_export1=yaml_content["ZCR"]["Carpetaok"]
ZCR_path_export2=yaml_content["ZCR"]["Carpetanok"]
FvsA_path_export1=yaml_content["FvsA"]["Carpetaok"]
FvsA_path_export2=yaml_content["FvsA"]["Carpetanok"]
SPECTROGRAM_path_export1=yaml_content["SPECTROGRAM"]["Carpetaok"]
SPECTROGRAM_path_export2=yaml_content["SPECTROGRAM"]["Carpetanok"]
GREYSPECTROGRAM_path_export1=yaml_content["GREYSPECTROGRAM"]["Carpetaok"]
GREYSPECTROGRAM_path_export2=yaml_content["GREYSPECTROGRAM"]["Carpetanok"]
MELSPECTROGRAM_path_export1=yaml_content["MELSPECTROGRAM"]["Carpetaok"]
MELSPECTROGRAM_path_export2=yaml_content["MELSPECTROGRAM"]["Carpetanok"]
CHROMAGRAM_path_export1=yaml_content["CHROMAGRAM"]["Carpetaok"]
CHROMAGRAM_path_export2=yaml_content["CHROMAGRAM"]["Carpetanok"]
MFCC_path_export1=yaml_content["MFCC"]["Carpetaok"]
MFCC_path_export2=yaml_content["MFCC"]["Carpetanok"]
DELTA_MFCC_path_export1=yaml_content["DELTA_MFCC"]["Carpetaok"]
DELTA_MFCC_path_export2=yaml_content["DELTA_MFCC"]["Carpetanok"]
DELTA2_MFCC_path_export1=yaml_content["DELTA2_MFCC"]["Carpetaok"]
DELTA2_MFCC_path_export2=yaml_content["DELTA2_MFCC"]["Carpetanok"]
BER_path_export1=yaml_content["BER"]["Carpetaok"]
BER_path_export2=yaml_content["BER"]["Carpetanok"]
SpecCent_path_export1=yaml_content["SpecCent"]["Carpetaok"]
SpecCent_path_export2=yaml_content["SpecCent"]["Carpetanok"]
Bandwidth_path_export1=yaml_content["Bandwidth"]["Carpetaok"]
Bandwidth_path_export2=yaml_content["Bandwidth"]["Carpetanok"]
SpecContrast_path_export1=yaml_content["SpecContrast"]["Carpetaok"]
SpecContrast_path_export2=yaml_content["SpecContrast"]["Carpetanok"]
SpecRollOff_path_export1=yaml_content["SpecRollOff"]["Carpetaok"]
SpecRollOff_path_export2=yaml_content["SpecRollOff"]["Carpetanok"]
PolyFeatures_path_export1=yaml_content["PolyFeatures"]["Carpetaok"]
PolyFeatures_path_export2=yaml_content["PolyFeatures"]["Carpetanok"]
Tonnetz_path_export1=yaml_content["Tonnetz"]["Carpetaok"]
Tonnetz_path_export2=yaml_content["Tonnetz"]["Carpetanok"]

FRAME_SIZE = yaml_content["Frame_size"]
HOP_LENGTH = yaml_content["Hop_lenght"]

class Features():
    #Waveform
    # Simple WAVEFORM to check clip trimming accuracy 
    def __init__(self, y,sr,res,archivo):
        
        self.y=y
        self.sr_size=sr
        self.res=res
        self.archivo=archivo
       
    def waveform(y,sr,res,archivo,path_day,path_actual):
        fig, ax = plt.subplots() 
        img = librosa.display.waveshow(y, sr=sr) 
        ax.set(title='WAVEFORM') 
        #The first strips off any trailing slashes, the second gives you the last part of the path. 
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual, WAVEFORM_path_export1,WAVEFORM_path_export2,res,'waveform',fig, archivo)
        plt.close()
        

    def amplitudeenvelope(y,sr,res,archivo,path_day,path_actual):
        
        # number of frames in amplitude envelope
        ae_y = Audico.fancy_amplitude_envelope(y, FRAME_SIZE, HOP_LENGTH)
        len(ae_y)

        #Visualizing amplitud envelope
        frames = range(len(ae_y))
        t = librosa.frames_to_time(frames,sr=sr, hop_length=HOP_LENGTH)

        fig, ax = plt.subplots()
        img=librosa.display.waveshow(y,sr=sr, alpha=0.5)
        plt.plot(t, ae_y, color="r")
        #plt.ylim((-1, 1))
        ax.set(title="Amplitude envelope")
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,AMPLITUDEENV_path_export1,AMPLITUDEENV_path_export2,res,'AmplitudEnvelope',fig,archivo)
        plt.close()

    def RootMeanSquaredError(y,sr,res,archivo,path_day,path_actual):
        
        rms_y = librosa.feature.rms(y=y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        #Visualise RMSE + waveform
        frames = range(len(rms_y))
        t = librosa.frames_to_time(frames,sr=sr, hop_length=HOP_LENGTH)
        # rms energy is graphed in red
        plt.figure(figsize=(15, 17))
        fig, ax = plt.subplots()
        librosa.display.waveshow(y,sr=sr, alpha=0.5)
        plt.plot(t, rms_y, color="r")
        #plt.ylim((-1, 1))
        ax.set(title="RMS energy")
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,RMSE_path_export1,RMSE_path_export2,res,'RMSE',fig,archivo)
        plt.close()

        return t

    def ZeroCrossingRate(y,sr,res,t,archivo,path_day,path_actual):
        
        #Zero-crossing rate with Librosa
        zcr_y = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        #Visualise zero-crossing rate with Librosa
        plt.figure(figsize=(15, 10))
        fig, ax = plt.subplots()
        plt.plot(t, zcr_y, color="r")
        plt.ylim(0, 1)
        ax.set(title="Zero Croosing Rate")
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,ZCR_path_export1,ZCR_path_export2,res,'ZCR',fig,archivo)
        plt.close()

    def FreqAmp(y,sr, res,archivo,path_day,path_actual):
        
        #Frequency vs amplitude graph
        fft=np.fft.fft(y)

        magnitude=np.abs(fft)
        frequency=np.linspace(0,sr,len(magnitude))

        left_frequency=frequency[:int(len(frequency)/2)]
        left_magnitude=magnitude[:int(len(frequency)/2)]

        fig, bx=plt.subplots()
        plt.plot(left_frequency,left_magnitude)
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        bx.set(title="Frequency vs Amplitude")
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,FvsA_path_export1,FvsA_path_export2,res,'FvsA',fig,archivo)
        plt.close()

    def Spectrogram(S_db, sr,res,archivo,path_day,path_actual):
        
        # SPECTROGRAM representation - object-oriented interface 
        plt.figure(figsize=(25, 10))
        fig, ax = plt.subplots() 
        img = librosa.display.specshow(S_db,sr=sr, x_axis='time', y_axis='linear') 
        img = librosa.display.specshow(S_db,sr=sr, x_axis='time', y_axis='log') 
        plt.colorbar(format="%+2.f")
        ax.set(title='SPECTROGRAM') 
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,SPECTROGRAM_path_export1,SPECTROGRAM_path_export2,res,'Spectrogram',fig,archivo)
        plt.close()

    def GreySpectrogram(S_db,sr, res,archivo,path_day,path_actual):
        
        # SPECTROGRAM representation - object-oriented interface 
        fig, ax = plt.subplots() 
        img = librosa.display.specshow(S_db,sr=sr, x_axis='time', y_axis='linear') 
        img = librosa.display.specshow(S_db,sr=sr, x_axis='time', y_axis='log', cmap='gray_r') 
        plt.colorbar(format="%+2.f")
        ax.set(title='GREY SPECTROGRAM') 
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,GREYSPECTROGRAM_path_export1,GREYSPECTROGRAM_path_export2,res,'Grey Spectrogram',fig,archivo)
        plt.close()

    def MelSpectrogram(y,sr, res,archivo,path_day,path_actual):
        
        #Extracting Mel Spectrogram n_fft=frame size
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=HOP_LENGTH, n_mels=90)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        plt.figure(figsize=(25, 10))
        fig, ax = plt.subplots() 
        img=librosa.display.specshow(log_mel_spectrogram, 
                                x_axis="time",
                                y_axis="log", 
                                sr=sr,hop_length=HOP_LENGTH)
        plt.colorbar(format="%+2.f")
        ax.set(title='MEL SPECTROGRAM') 
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,MELSPECTROGRAM_path_export1,MELSPECTROGRAM_path_export2,res,'MelSpectrogram',fig,archivo)
        plt.close()
        
    def Chromagram(y,sr,res,archivo,path_day,path_actual):
        
        #CHROMAGRAM representation - object-oriented interface 
        CHROMAGRAM = librosa.feature.chroma_cqt(y=y, sr=sr,fmin=70) 
        fig, ax = plt.subplots() 
        img = librosa.display.specshow(CHROMAGRAM,sr=sr, y_axis='chroma', x_axis='time') 
        plt.colorbar(format="%+2.f")
        ax.set(title='CHROMAGRAM') 
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,CHROMAGRAM_path_export1,CHROMAGRAM_path_export2,res,'Chromogram',fig,archivo)
        plt.close()

    def MFCCs(y,sr,res,archivo,path_day,path_actual):
        
        #MFCC representation - object-oriented interface 
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048) #n_fft=1200
        fig, ax = plt.subplots() 
        img = librosa.display.specshow(mfccs,sr=sr, x_axis='time') 
        plt.colorbar(format="%+2.f")
        ax.set(title='Mel-frequency cepstral coefficients (MFCCs)') 
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,MFCC_path_export1,MFCC_path_export2,res,'MFCCs',fig,archivo)
        plt.close()

        return mfccs

    def DeltaMFCCs(mfccs,sr,res,archivo,path_day,path_actual):
        
        delta_mfccs = librosa.feature.delta(mfccs)

        plt.figure() #figsize=(25, 10)
        fig, ax = plt.subplots() 
        img = librosa.display.specshow(delta_mfccs, x_axis='time',sr=sr) 
        plt.colorbar(format="%+2.f")
        ax.set(title='Delta Mel-frequency cepstral coefficients (MFCCs)') 
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,DELTA_MFCC_path_export1,DELTA_MFCC_path_export2,res,'DeltaMFCCs',fig,archivo)
        plt.close()

    def Delta2MFCCs(mfccs,sr,res,archivo,path_day,path_actual):
        
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        plt.figure() #figsize=(25, 10)
        fig, ax = plt.subplots() 
        img = librosa.display.specshow(delta2_mfccs, x_axis='time',sr=sr) 
        plt.colorbar(format="%+2.f")
        ax.set(title='Delta2 Mel-frequency cepstral coefficients (MFCCs)') 
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,DELTA2_MFCC_path_export1,DELTA2_MFCC_path_export2,res,'Delta2MFCCs',fig,archivo)
        plt.close()

    def BandEnergyRatio(y,sr,res,archivo,path_day,path_actual):
        
        #HOP_SIZE=512
        y_spec = librosa.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)

        split_frequency_bin =Audico.calculate_split_frequency_bin(2000, sr, 1025)
        

        ber_y =ratio.band_energy_ratio (y_spec, 2000, sr=sr)
        #Visualise Band Energy Ratio
        frames = range(len(ber_y))
        t = librosa.frames_to_time(frames,sr=sr, hop_length=HOP_LENGTH)

        plt.figure(figsize=(25, 10))
        fig, ax = plt.subplots()
        plt.plot(t, ber_y, color="b")
        ax.set(title="Band Energy Ratio")
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,BER_path_export1,BER_path_export2,res,'Band Energy Ratio',fig,archivo)
        plt.close()

    def SpectralCentroid(y,sr,t,res,archivo,path_day,path_actual):
        sc_y = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        #Visualising spectral centroid
        plt.figure(figsize=(25,10))
        fig, ax = plt.subplots()
        plt.plot(t, sc_y, color='b')
        ax.set(title="Spectral Centroid")
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,SpecCent_path_export1,SpecCent_path_export2,res,'Spectral Centroid',fig,archivo)
        plt.close()

    def Bandwidht(y,sr,t,res,archivo,path_day,path_actual):
        
        #Spectral bandwidth with Librosa
        ban_y = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        #Visualising spectral bandwidth
        plt.figure(figsize=(25,10))
        fig, ax = plt.subplots()
        plt.plot(t, ban_y, color='b')
        ax.set(title="Bandwidth")
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,Bandwidth_path_export1,Bandwidth_path_export2,res,'Bandwidth',fig,archivo)
        plt.close()

    def SpectralContrast(y,sr,res,archivo,path_day,path_actual):
        
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        img1 = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                        ref=np.max),
                                y_axis='log', x_axis='time', ax=ax[0],sr=sr)
        fig.colorbar(img1, ax=[ax[0]], format='%+2.0f dB')
        ax[0].set(title='Power spectrogram')
        ax[0].label_outer()
        img2 = librosa.display.specshow(contrast, x_axis='time', ax=ax[1],sr=sr)
        fig.colorbar(img2, ax=[ax[1]])
        ax[1].set(ylabel='Frequency bands', title='Spectral contrast')
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,SpecContrast_path_export1,SpecContrast_path_export2,res,'Spectral Contrast',fig,archivo)
        plt.close()

    def SpectralRollOff(y,S_db,sr,res,archivo,path_day,path_actual):
        # Spectral Flatness
        #From time-series input
        flatness = librosa.feature.spectral_flatness(y=y)
        #From spectrogram input
        S, phase = librosa.magphase(librosa.stft(y))
        librosa.feature.spectral_flatness(S=S)
        #From power spectrogram input
        S_power = S ** 2
        librosa.feature.spectral_flatness(S=S_power, power=1.0)
        #Spectral RollOff
        # Approximate maximum frequencies with roll_percent=0.85 (default)
        librosa.feature.spectral_rolloff(y=y, sr=sr)

        # Approximate maximum frequencies with roll_percent=0.99
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99,hop_length=HOP_LENGTH)
        
        # Approximate minimum frequencies with roll_percent=0.01
        rolloff_min = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.01,hop_length=HOP_LENGTH)
        

        fig, ax = plt.subplots()
        librosa.display.specshow(S_db,sr=sr, y_axis='log', x_axis='time', ax=ax)
        ax.plot(librosa.times_like(rolloff,sr=sr,hop_length=HOP_LENGTH), rolloff[0], label='Roll-off frequency (0.99)')
        ax.plot(librosa.times_like(rolloff,sr=sr,hop_length=HOP_LENGTH), rolloff_min[0], color='w',
                label='Roll-off frequency (0.01)')
        ax.legend(loc='lower right')
        ax.set(title='log Power spectrogram')
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,SpecRollOff_path_export1,SpecRollOff_path_export2,res,'Spectral Rolloff',fig,archivo)
        plt.close()

        return S

    def PolyFeatures(S,sr,res,archivo,path_day,path_actual):
        
        p0 = librosa.feature.poly_features(S=S, sr=sr, order=0)
        p1 = librosa.feature.poly_features(S=S, sr=sr,order=1)
        p2 = librosa.feature.poly_features(S=S, sr=sr,order=2)

        fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8, 8))
        times = librosa.times_like(p0,sr=sr)
        ax[0].plot(times, p0[0], label='order=0', alpha=0.8)
        ax[0].plot(times, p1[1], label='order=1', alpha=0.8)
        ax[0].plot(times, p2[2], label='order=2', alpha=0.8)
        ax[0].legend()
        ax[0].label_outer()
        ax[0].set(ylabel='Constant term ')
        ax[1].plot(times, p1[0], label='order=1', alpha=0.8)
        ax[1].plot(times, p2[1], label='order=2', alpha=0.8)
        ax[1].set(ylabel='Linear term')
        ax[1].label_outer()
        ax[1].legend()
        ax[2].plot(times, p2[0], label='order=2', alpha=0.8)
        ax[2].set(ylabel='Quadratic term')
        ax[2].legend()
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                y_axis='log', x_axis='time', ax=ax[3],sr=sr)
        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,PolyFeatures_path_export1,PolyFeatures_path_export2,res,'Poly features',fig,archivo)
        plt.close()

    def Tonnetz(y,sr,res,archivo,path_day,path_actual):
        
        y = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr,fmin=72)
        
        fig, ax = plt.subplots(nrows=2, sharex=True)
        img1 = librosa.display.specshow(tonnetz,
                                        y_axis='tonnetz', x_axis='time', ax=ax[0],sr=sr)
        ax[0].set(title='Tonal Centroids (Tonnetz)')
        ax[0].label_outer()
        img2 = librosa.display.specshow(librosa.feature.chroma_cqt(y=y, sr=sr,fmin=72),
                                        y_axis='chroma', x_axis='time', ax=ax[1],sr=sr)
        ax[1].set(title='Chroma')
        fig.colorbar(img1, ax=[ax[0]])
        fig.colorbar(img2, ax=[ax[1]])

        ClaseAudio2.CargaeImagenAudio.guardarimagen(path_day,path_actual,Tonnetz_path_export1,Tonnetz_path_export2,res,'Tonnetz',fig,archivo)
        plt.close()

class Audico(Features):
    def __init__(self, y, sr, frame_size, hop_lenght,split_frequency,num_frequency_bins):
        
        Features.__init__(self, y, sr)
        self.frame_size=frame_size
        self.hop_lenght=hop_lenght
        self.split_frequency=split_frequency
        self.num_frequency_bins=num_frequency_bins

    def fancy_amplitude_envelope(y, frame_size, hop_length):
        """Fancier Python code to calculate the amplitude envelope of a signal with a given frame size."""
        return np.array([max(y[i:i+frame_size]) for i in range(0, len(y), hop_length)])
    
    def calculate_split_frequency_bin(split_frequency, sr, num_frequency_bins):
        """Infer the frequency bin associated to a given split frequency."""
        
        frequency_range = sr / 2
        frequency_delta_per_bin = frequency_range / num_frequency_bins
        split_frequency_bin = math.floor(split_frequency / frequency_delta_per_bin)
        return int(split_frequency_bin)

class ratio(Audico):
    def __init__(self,spectrogram, split_frequency,sr):
        Features.__init__(self,sr)
        Audico.__init__(self,split_frequency)
        self.spectrogram=spectrogram
        
    def band_energy_ratio(spectrogram, split_frequency, sr):
        #Calculate band energy ratio with a given split frequency.
        
        split_frequency_bin = Audico.calculate_split_frequency_bin(split_frequency, sr, len(spectrogram[0]))
        band_energy_ratio = []
        
        # calculate power spectrogram
        power_spectrogram = np.abs(spectrogram) ** 2
        power_spectrogram = power_spectrogram.T
        
        # calculate BER value for each frame
        for frame in power_spectrogram:
            sum_power_low_frequencies = frame[:split_frequency_bin].sum()
            sum_power_high_frequencies = frame[split_frequency_bin:].sum()
            band_energy_ratio_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
            band_energy_ratio.append(band_energy_ratio_current_frame)
        
        return np.array(band_energy_ratio)
