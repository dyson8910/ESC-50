#!/usr/bin/env python3
# encoding: utf-8
import numpy as np
import soundfile as sf
import pandas as pd
import os,datetime,argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

fs = 44100

def hz2mel(f):
    return 1127.01048*np.log(f/700.0+1.0)
def mel2hz(m):
    return 700*(np.exp(m/1127.01048)-1.0)
def melFilterBank(fs,fmin,fmax,length,numChannels):
    melmax = hz2mel(fmax)
    melmin = hz2mel(fmin)
    nmax = length / 2
    df = fs / float(length)
    dmel = (melmax - melmin) / (numChannels + 1.0)
    melcenters = np.arange(0,numChannels + 2) * dmel + melmin
    fcenters = mel2hz(melcenters)
    index = np.round(fcenters/df)
    indexstart = index[:numChannels].copy()
    indexcenter = index[1:numChannels+1].copy()
    indexstop = index[2:].copy()
    filterbank = np.zeros((numChannels,int(nmax)))
    for c in range(numChannels):
        increment = 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c],indexcenter[c]):
            filterbank[int(c),int(i)] = (i-indexstart[c]) * increment
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c],indexstop[c]):
            filterbank[int(c),int(i)] = 1.0 - ((i - indexcenter[c]) * decrement)
    return filterbank,fcenters[1:numChannels+1]

def generate_data(filename,mels,ffts,shift,n,time):
    filterbank, fc = melFilterBank(fs,0,fs//2,ffts,mels)
    suffix = ['','_noise','_stretch0','_stretch1','_stretch2','_stretch3']
    l = 8820
    data = np.array([])
    logmel = np.array([])
    frequency = np.zeros(ffts//2)
    window = np.hamming(ffts)
    for s in suffix:
        wave,rate = sf.read('../augmentation/'+filename+s+'.wav')
        # extraction
        amp = np.abs(wave)
        amp_thre = np.max(amp) * 0.08
        idx = amp > amp_thre
        index = []
        sidx = np.arange(wave.shape[0])[idx][0]
        tidx = sidx
        while True:
            idx = amp[tidx+1:tidx+l] > amp_thre
            diff = np.arange(idx.shape[0])[idx]
            if diff.shape[0] != 0:
                tidx += diff[-1] + 1
            else:
                fidx = tidx
                index.append([np.max([sidx-l//2,0]),np.min([fidx+l//2,wave.shape[0]-1])])
                idx = amp[fidx+1:] > amp_thre
                diff = np.arange(idx.shape[0])[idx]
                if diff.shape[0] != 0:
                    sidx = fidx + 1 + diff[0]
                    tidx = sidx
                else:
                    break
        index = np.array(index,dtype=int)
        # zero padding and characteristic frequency extraction
        waves = []
        zerolen = int(time*shift*fs/1000.)
        for i in index:
            waves.append(np.append(np.append(np.zeros(zerolen//2),wave[i[0]:i[1]]),np.zeros(zerolen//2)))
            if s != '':
                continue
            spectrum = []
            for t in np.arange(int((waves[-1].shape[0]-ffts)/(shift*fs/1000.))):
                spec = np.fft.fft(window*waves[-1][int(t*shift*fs/1000.):int(t*shift*fs/1000.)+ffts])[:ffts//2]
                spectrum.append(np.abs(spec))
            spectrum = np.array(spectrum)
            spectrum = np.sum(spectrum,axis=0)
            frequency += spectrum > np.max(spectrum)*0.1
        # generate dataset
        indexs = np.random.choice(np.arange(index.shape[0]),n,replace=True)
        d = []
        logmel_ = []
        for i in indexs:
            sidx = np.random.choice(np.arange(waves[i].shape[0]-zerolen-1-ffts),1)[0]
            w = waves[i][sidx:sidx+int(time*shift*fs/1000.)+ffts]
            d.append(w)
            mel = []
            delta = []
            for t in range(time):
                s1 = np.fft.fft(w[int(t*shift*fs/1000.):int(t*shift*fs/1000.)+ffts])[:ffts//2]
                s2 = np.fft.fft(w[int((t+1)*shift*fs/1000.):int((t+1)*shift*fs/1000.)+ffts])[:ffts//2]
                m1 = np.log10(np.dot(filterbank,np.abs(s1)+np.ones(ffts//2)*10**(-16)))
                m2 = np.log10(np.dot(filterbank,np.abs(s2)+np.ones(ffts//2)*10**(-16)))
                mel.append(m2)
                delta.append(m2-m1)
            mel = np.array(mel)
            delta =np.array(delta)
            logmel_.append(np.array([mel,delta]))
        d = np.array(d)
        logmel_ = np.array(logmel_)
        if data.shape[0] == 0:
            data = d
            logmel = logmel_
        else:
            data = np.concatenate([data,d])
            logmel = np.concatenate([logmel,logmel_])
    return data, frequency>0, logmel

def main():
    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--mels','-m',type=int,default=128,help='mels')
    parser.add_argument('--ffts','-f',type=int,default=2048,help='fft_length[sample]')
    parser.add_argument('--shift','-s',type=float,default=16.,help='shift time[ms]')
    parser.add_argument('--n_data','-n',type=int,default=10,help='number of data for 1 wav file')
    parser.add_argument('--time','-t',type=int,default=26,help='length of data')
    args = parser.parse_args()
    datasetpath = '../dataset/'+datetime.datetime.now().strftime('%Y%m%d%H%M/')
    if not os.path.exists(datasetpath):
        os.makedirs(datasetpath)
    metadata = pd.read_csv('../meta/esc50.csv')
    fold = np.arange(5)+1
    for f in fold:
        print('start generate dataset fold=={}'.format(f))
        meta = metadata[metadata['fold']==f]
        meta = meta.sort_values('target').reset_index(drop=True)
        dataset = np.array([])
        logmel = np.array([])
        label = np.array([])
        frequency = {}
        bar = tqdm(range(meta.shape[0]))
        for i in bar:
            m = meta.loc[i]
            filename = m['filename'][:-4]
            bar.set_description('Processing {}'.format(filename))
            data, frequency_, logmel_ = generate_data(filename,args.mels,args.ffts,args.shift,args.n_data,args.time) # shape(n_data*6,wave_length), (ffts//2,)
            label_ = np.full(data.shape[0],m['target'])
            if dataset.shape[0] == 0:
                dataset = data
                logmel = logmel_
                label = label_
            else:
                dataset = np.concatenate([dataset,data]) # shape (dataset_size, wave_length)
                logmel = np.concatenate([logmel,logmel_])
                label = np.concatenate([label,label_])
            if m['target'] in frequency:
                frequency[m['target']] += frequency_
            else:
                frequency[m['target']] = frequency_
        np.save(datasetpath+'fold{}dataset'.format(f),dataset)
        np.save(datasetpath+'fold{}logmel'.format(f),logmel)
        np.save(datasetpath+'fold{}label'.format(f),label)
        np.save(datasetpath+'fold{}frequency'.format(f),frequency) # np.load(~,allow_pickle=True).item()

if __name__ == '__main__':
    main()
