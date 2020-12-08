#!/usr/bin/env python3
import numpy as np
import soundfile as sf
import pandas as pd
import librosa,os
from tqdm import tqdm

def add_noise(x, amp=0.001):
    return x+amp*np.random.randn(x.shape[0])

def stretch(x, rate=1.0):
    return librosa.effects.time_stretch(x, rate=rate)

def main():
    augdir = '../augmentation/'
    if not os.path.exists(augdir):
        os.makedirs(augdir)
    meta = pd.read_csv('../meta/esc50.csv')
    num = meta.shape[0]
    bar = tqdm(range(num))
    for i in bar:
        data = meta.loc[i]
        filename = data['filename']
        bar.set_description('Processing {}'.format(filename))
        wave, rate = sf.read('../audio/'+filename)
        if(rate != 44100):
            continue
        sf.write(augdir+filename, wave, rate)
        m = np.max(np.abs(wave))
        wave_noise = add_noise(wave, m*0.01)
        sf.write(augdir+filename[:-4]+'_noise.wav', wave_noise, rate)
        for j,t in enumerate([0.95,0.975,1.025,1.05]):
            wave_s = stretch(wave,t)
            sf.write(augdir+filename[:-4]+'_stretch{}.wav'.format(j), wave_s, rate)

if __name__ == '__main__':
    main()
