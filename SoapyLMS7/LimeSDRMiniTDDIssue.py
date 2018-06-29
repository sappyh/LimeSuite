#!/usr/bin/env python3

import sys
sys.path.append('/usr/local/lib/python3/dist-packages')
import SoapySDR
from SoapySDR import * #constants
import numpy as np
from PlotUtils import *
import time
import os

BW = 10e6
RATE = 5e6
FREQ = 2.9e9
WAVE = 0.3e6
NSAMPS = 1024

def main():
    sdr = SoapySDR.Device(dict(driver='lime'))
    for ch in [0, ]:
        sdr.setSampleRate(SOAPY_SDR_TX, ch, RATE)
        sdr.setSampleRate(SOAPY_SDR_RX, ch, RATE)
        sdr.setAntenna(SOAPY_SDR_TX, ch, "BAND1")
        sdr.setAntenna(SOAPY_SDR_RX, ch, "LNAH")
        sdr.setBandwidth(SOAPY_SDR_RX, ch, BW)
        sdr.setBandwidth(SOAPY_SDR_TX, ch, BW)
        sdr.writeSetting(SOAPY_SDR_TX, ch, 'TSP_CONST', str(1 << 13))

    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0,])
    figWave = LocalFigure(size=[4, 2], title="LimeSDR Mini TDD demo [left column Tx LO and Rx LO active] [right column: shared LO TDD mode]")
    figFreq = LocalFigure(size=[4, 2], title="LimeSDR Mini TDD demo [left column Tx LO and Rx LO active] [right column: shared LO TDD mode]")

    for i, (freq, iamp, pad, tia, lna) in enumerate([
            (3.1e9, 0, 45, 12, 20),
            (3.4e9, 12, 45, 12, 20),
            (3.5e9, 12, 45, 12, 20),
            (3.6e9, 12, 45, 12, 27)]):
        print('========= %g GHz ========'%(freq/1e9))
        sdr.setFrequency(SOAPY_SDR_RX, ch, freq)
        sdr.setFrequency(SOAPY_SDR_TX, ch, freq+WAVE)
        sdr.setFrequency(SOAPY_SDR_TX, ch, "BB", 0)
        sdr.setGain(SOAPY_SDR_TX, ch, "IAMP", iamp)
        sdr.setGain(SOAPY_SDR_TX, ch, "PAD", pad)
        sdr.setGain(SOAPY_SDR_RX, ch, "LNA", lna)
        sdr.setGain(SOAPY_SDR_RX, ch, "TIA", tia)
        sdr.writeSetting(SOAPY_SDR_TX, ch, 'CALIBRATE', str(BW))
        sdr.writeSetting(SOAPY_SDR_RX, ch, 'CALIBRATE', str(BW))

        sdr.activateStream(rxStream, SOAPY_SDR_END_BURST, 0, NSAMPS)
        buffs0 = np.zeros(NSAMPS, np.complex64)
        r = sdr.readStream(rxStream, [buffs0], NSAMPS)
        print(r)
        sdr.deactivateStream(rxStream)

        ps = LogPowerFFT(buffs0)
        figWave.addTimePlot(i*2+1, traces=[buffs0], rate=RATE, title='LO tuned with offset %g GHz'%(freq/1e9))
        figFreq.addFreqPlot(i*2+1, traces=[ps], rate=RATE, title='LO tuned with offset %g GHz'%(freq/1e9))

        #now use tdd mode
        sdr.setFrequency(SOAPY_SDR_RX, ch, freq)
        sdr.setFrequency(SOAPY_SDR_TX, ch, freq)
        sdr.setFrequency(SOAPY_SDR_TX, ch, "BB", WAVE)
        sdr.writeSetting(SOAPY_SDR_TX, ch, 'CALIBRATE', str(BW))
        sdr.writeSetting(SOAPY_SDR_RX, ch, 'CALIBRATE', str(BW))

        sdr.activateStream(rxStream, SOAPY_SDR_END_BURST, 0, NSAMPS)
        buffs0 = np.zeros(NSAMPS, np.complex64)
        r = sdr.readStream(rxStream, [buffs0], NSAMPS)
        print(r)
        sdr.deactivateStream(rxStream)

        ps = LogPowerFFT(buffs0)
        figWave.addTimePlot(i*2+2, traces=[buffs0], rate=RATE, title='LO tuned in TDD mode %g GHz'%(freq/1e9))
        figFreq.addFreqPlot(i*2+2, traces=[ps], rate=RATE, title='LO tuned in TDD mode %g GHz'%(freq/1e9))

    freqOut = '/tmp/freq.png'
    waveOut = '/tmp/wave.png'
    figWave.savefig(waveOut)
    figFreq.savefig(freqOut)
    figWave.close()
    figFreq.close()
    print('-------->    '+freqOut)
    print('-------->    '+waveOut)
    print('Done!')

if __name__ == '__main__': main()
