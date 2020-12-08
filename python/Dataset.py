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
    filterbank,fc = melFilterBank(fs,0,fs/2,ffts,mels)
    suffix = ['','_noise','_stretch0','_stretch1','_stretch2','_stretch3']
    l = 8820
    data = np.array([])
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
        # zero padding
        waves = []
        zerolen = int((time)*shift*fs/1000.)
        for i in index:
            waves.append(np.append(np.append(np.zeros(zerolen//2),wave[i[0]:i[1]]),np.zeros(zerolen//2)))
        # generate dataset
        indexs = np.random.choice(np.arange(index.shape[0]),n,replace=True)
        d = []
        for i in indexs:
            sidx = np.random.choice(np.arange(waves[i].shape[0]-zerolen-1-ffts),1)[0]
            if sidx < ffts:
                w = np.append(np.zeros(ffts-sidx),waves[i])
            else:
                w = waves[i][sidx:]
            mel = []
            delta = []
            for t in np.arange(time):
                s1 = np.fft.fft(w[int(t*shift*fs/1000.):int(t*shift*fs/1000.)+ffts])
                s2 = np.fft.fft(w[int((t+1)*shift*fs/1000.):int((t+1)*shift*fs/1000.)+ffts])
                m1 = np.log10(np.dot(filterbank, np.abs(s1[:ffts//2])+np.ones(ffts//2)*10**(-16)))
                m2 = np.log10(np.dot(filterbank, np.abs(s2[:ffts//2])+np.ones(ffts//2)*10**(-16)))
                mel.append(m2)
                delta.append(m2-m1)
            mel = np.array(mel)
            delta = np.array(delta)
            mel = mel/np.linalg.norm(mel)
            delta = delta/np.linalg.norm(delta)
            d.append(np.array([mel,delta]))
        d = np.array(d)
        if data.shape[0] == 0:
            data = d
        else:
            data = np.concatenate([data,d])
    return data

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
        label = np.array([])
        bar = tqdm(range(meta.shape[0]))
        for i in bar:
            m = meta.loc[i]
            filename = m['filename'][:-4]
            bar.set_description('Processing {}'.format(filename))
            data = generate_data(filename,args.mels,args.ffts,args.shift,args.n_data,args.time) # shape(n_data*6,2,mels,time)
            label_ = np.full(data.shape[0],m['target'])
            if dataset.shape[0] == 0:
                dataset = data
                label = label_
            else:
                dataset = np.concatenate([dataset,data]) # shape (dataset_size, 2, mels, time)
                label = np.concatenate([label,label_])
        np.save(datasetpath+'fold{}dataset'.format(f),dataset)
        np.save(datasetpath+'fold{}label'.format(f),label)

if __name__ == '__main__':
    main()
