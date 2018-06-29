########################################################################
## Utility functions to take FFTs and locate tones
########################################################################

import numpy as np
import math
import scipy.signal

def LogPowerFFT(samps, numBins=None, peak=1.0, reorder=True, window=None):
    """
    Calculate the log power FFT bins of the complex samples.
    When numBins is not specified, the FFT is across all samples.
    @param samps numpy array of complex samples
    @param numBins take FFTs of this size and average their bins
    @param peak maximum value of a sample (floats are usually 1.0, shorts are 32767)
    @param reorder True to reorder so the DC bin is in the center
    @param window function or None for default flattop window
    @return an array of real values FFT power bins
    """

    #support complex integers (2d arrays)
    if hasattr(samps, 'shape') and len(samps.shape) > 1 and samps.shape[1] == 2:
        scale = 1 << ((samps.dtype.itemsize*8)-1)
        samps = np.array(list(map(lambda x: complex(*x), samps)), np.complex64)/scale

    size = len(samps)

    if numBins is not None:
        numFFTs = int(size/numBins)
        #print 'size', size
        #print 'numBins', numBins
        #print 'numFFTs', numFFTs
        allBins = list()
        for i in range(numFFTs):
            bins = LogPowerFFT(samps[i*numBins:(i+1)*numBins], peak=peak, reorder=reorder)
            allBins.append(np.exp(bins))
        avgBins = sum(allBins)/numFFTs
        return np.log(avgBins)

    #scale by dividing out the peak, full scale is 0dB
    scaledSamps = samps/peak

    #calculate window
    if not window: window = scipy.signal.hann
    windowBins = window(size)
    windowPower = math.sqrt(sum(windowBins**2)/size)

    #apply window
    windowedSamps = np.multiply(windowBins, scaledSamps)

    #window and fft gain adjustment
    gaindB = 20*math.log10(size) + 20*math.log10(windowPower)

    #take fft
    fftBins = np.abs(np.fft.fft(windowedSamps))
    fftBins = np.maximum(fftBins, 1e-20) #clip
    powerBins = 20*np.log10(fftBins) - gaindB

    #bin reorder
    if reorder:
        idx = np.argsort(np.fft.fftfreq(len(powerBins)))
        powerBins = powerBins[idx]
        #even fft has two DC bins, just remove other DC bin for now
        #if (len(powerBins) % 2) == 0: powerBins = powerBins[1:]

    return powerBins

def MeasureTonePower(powerBins, sampRate, toneFreq, delta=10):
    """
    Measure a tone level by looking for max power at a given position.
    """

    tonePosition = len(powerBins)*(toneFreq+sampRate/2)/sampRate
    startIndex = max(0, int(tonePosition)-delta)
    stopIndex = min(len(powerBins), int(tonePosition)+delta) #non inclusive

    maxPowerIndex = startIndex
    for i in range(startIndex, stopIndex):
        if powerBins[i] > powerBins[maxPowerIndex]:
            maxPowerIndex = i

    return powerBins[maxPowerIndex]

########################################################################
## We are going to plot a lot of waveforms to file
## This is basically a plot styler to avoid code duplication,
## and to create a common style for generating the plots.
########################################################################

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import queue
import threading
import logging
import traceback
import io
import atexit

class LocalFigure(object):

    def __init__(self, size=[1, 1], title=None):
        self.fig = plt.figure(figsize=(20, 8), dpi=100)
        self.nrows = size[0]
        self.ncols = size[1]
        if title is not None:
            self.fig.suptitle(title, fontsize=16, horizontalalignment='left', x=0.01)
        self.fig.subplots_adjust(hspace=.5, top=.85)

    def addTimePlot(self, axnum, traces, rate, title=None, xlabel='Time (ms)', ylabel='Amplitude (units)', xscale=1e-3, ylim=None):
        ax = self.fig.add_subplot(self.nrows, self.ncols, axnum)  # specify (nrows, ncols, axnum)
        ax.grid(True)
        if title: ax.set_title(title, fontsize=10)
        if xlabel: ax.set_xlabel(xlabel, fontsize=10)
        if ylabel: ax.set_ylabel(ylabel, fontsize=10)

        for trace in traces:
            numSamps = len(trace)
            timeScale = np.arange(0, numSamps/(rate*xscale), 1/(rate*xscale))[:numSamps]

            if hasattr(trace, 'shape') and len(trace.shape) > 1 and trace.shape[1] == 2:
                ax.plot(timeScale, [x[0] for x in trace])
                ax.plot(timeScale, [x[1] for x in trace])
                ax.set_ylim(-(1<<15), +(1<<15))
            elif np.iscomplex(trace[0]):
                ax.plot(timeScale, np.real(trace))
                ax.plot(timeScale, np.imag(trace))
                ax.set_ylim(-1.0, +1.0)
            else:
                ax.plot(timeScale, trace)
                ax.set_ylim(-1.0, +1.0)

        if ylim is not None: ax.set_ylim(*ylim)

    def addFreqPlot(self, axnum, traces, rate, title=None,
        xlabel='Freq (MHz)', ylabel='Power (dBfs)',
        xscale=1e6, refLvl=0.0, plotNoise=True, centerFreq=0.0, dynRange=100,
    ):
        ax = self.fig.add_subplot(self.nrows, self.ncols, axnum)  # specify (nrows, ncols, axnum)
        ax.grid(True)
        if title: ax.set_title(title, fontsize=10)
        if xlabel: ax.set_xlabel(xlabel, fontsize=10)
        if ylabel: ax.set_ylabel(ylabel, fontsize=10)
        minNoise = refLvl-dynRange

        for ps in traces:
            noise, peaks = LogPowerFFTAnalysis(ps, rate)
            #pick the N largest tones above some threshold
            peaks = reversed(sorted(peaks, key=lambda x: x[1]))
            peaks = [p for p in peaks if p[1] > noise + 30]
            for freq, power in peaks[:5]:
                label = '%.3g\n%.3g'%(freq/xscale, power)
                x = int(ps.size*freq/rate) - ps.size/2
                if freq > 0.0:
                    xoff = 1.0
                    ha = 'left'
                else:
                    xoff = -1.0
                    ha = 'right'
                if power < (refLvl-30):
                    yoff = 1.0
                    va = 'bottom'
                else:
                    yoff = -1.0
                    va = 'top'
                ax.annotate(label, fontsize=8,
                    xy = (freq/xscale, power), xytext = (freq/xscale+xoff, power+yoff),
                    horizontalalignment=ha, verticalalignment=va,
                    xycoords = 'data', textcoords = 'data',
                    arrowprops = dict(arrowstyle = '->'))
                ax.scatter(freq/xscale, power, c='yellow', s=50)
            ax.plot(np.arange((centerFreq-rate/2)/xscale, (centerFreq+rate/2)/xscale, rate/len(ps)/xscale)[:len(ps)], ps, 'b')
            if plotNoise: ax.plot([-rate/2/xscale, rate/2/xscale], [noise, noise], 'r')
            minNoise = min(minNoise, noise)
            ax.set_ylim(minNoise-10, refLvl)

        #10 dB step size
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end, 10))

    def addDataPlot(self, axnum, xaxis, traces, title=None, xlabel=None, ylabel=None, ylim=None):
        ax = self.fig.add_subplot(self.nrows, self.ncols, axnum)  # specify (nrows, ncols, axnum)
        ax.grid(True)
        if title: ax.set_title(title, fontsize=10)
        if xlabel: ax.set_xlabel(xlabel, fontsize=10)
        if ylabel: ax.set_ylabel(ylabel, fontsize=10)
        if ylim: ax.set_ylim(*ylim)

        for i, trace in enumerate(traces):
            label = 'Trace %d'%i
            if isinstance(trace, tuple) and len(trace) == 2:
                trace, label = trace
            ax.plot(xaxis, trace, label=label)

        ax.legend(fontsize=10)

    def savefig(self, filepath):
        self.fig.savefig(filepath)

    def close(self):
        plt.close(self.fig)


def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = np.arange(len(v))
    
    v = np.asarray(v)
    
    if len(v) != len(x):
        raise Exception('Input vectors v and x must have same length')
    
    if not np.isscalar(delta):
        raise Exception('Input argument delta must be a scalar')
    
    if delta <= 0:
        raise Exception('Input argument delta must be positive')
    
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    
    lookformax = True
    
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
 
    return np.array(maxtab), np.array(mintab)


def LogPowerFFTAnalysis(powerBins, sampRate, threshold=10):
    """
    Analyze the power spectrum bins for noise floor and tones.
    @param powerBins an array of real values FFT power bins
    @param sampRate the sample rate for the FFT samples
    @threshold how many dB above the noise floor to detect a tone
    @return noiseFloor, [(tone0Freq, tone0Power), (tone1Freq, tone1Power), ]
    """

    #detect local minima and maxima
    maxtab, mintab = peakdet(powerBins, 20)

    #noise floor guess based on average of the log bins
    noiseFloor = np.mean(powerBins)

    #filter out peaks that are below the noise floor
    peaks = list()
    for i, v in maxtab:
        if v < noiseFloor + threshold: continue
        freq = (sampRate*i)/len(powerBins) - sampRate/2
        peaks.append((freq, v))

    return noiseFloor, peaks
