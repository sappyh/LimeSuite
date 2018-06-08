#!/usr/bin/env python3

import sys
sys.path.append('/usr/local/lib/python3/dist-packages')
import SoapySDR
from SoapySDR import * #constants
import numpy as np
from PlotUtils import *
import time
import os

FREQ_STEP = 100e6
FREQS = np.concatenate([
    np.arange(500e6, 1e9, FREQ_STEP),
    np.arange(1e9, 2e9, FREQ_STEP),
    np.arange(2e9, 3e9, FREQ_STEP),
    np.arange(3e9, 3.6e9, FREQ_STEP),
])
TX_LO_OFS = 5.3e6
TX_NCO = 2.1e6
SAMP_RATE = 20e6
BW = 30e6
FFT_SIZE = 2048
RX_DC_SETTLE = 0.3 #seconds is enough
PAD = 20.0
LNA = 30.0

CAL_PAD_LEVELS = dict()

def getRxPowerSpectrum(sdr, rxStream, fftSize):
    sdr.activateStream(rxStream, SOAPY_SDR_END_BURST, 0, fftSize)
    rxSamps = np.zeros(fftSize, np.complex64)
    r = sdr.readStream(rxStream, [rxSamps], fftSize, timeoutUs=1000000)
    if r.ret == SOAPY_SDR_TIMEOUT: raise Exception('readStream bug, got timeout!')
    try: assert(r.ret == fftSize)
    except: print(r); raise
    if sdr.getDriverKey() != 'iris':
        sdr.deactivateStream(rxStream) #FIXME limesuite: not automatic w/ end burst in limesdr
    return LogPowerFFT(rxSamps)

def collectSweepData(argsStr, channel=0, calibrate=False, dumpDir=None):
    sdr = SoapySDR.Device(argsStr)
    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [channel], dict(skipCal='true'))
    rows = list()
    hwInfo = sdr.getHardwareInfo()
    print(sdr.getHardwareKey())
    print(hwInfo)

    #constant settings, setup once
    if sdr.getDriverKey() == 'iris':
        sdr.setAntenna(SOAPY_SDR_RX, channel, "RX")
        sdr.setAntenna(SOAPY_SDR_TX, channel, "TRX")
    elif sdr.getHardwareKey() == 'LimeSDR-USB':
        sdr.setAntenna(SOAPY_SDR_RX, channel, "LNAH")
        sdr.setAntenna(SOAPY_SDR_TX, channel, "BAND2")
    sdr.setSampleRate(SOAPY_SDR_RX, channel, SAMP_RATE)
    sdr.setSampleRate(SOAPY_SDR_TX, channel, SAMP_RATE)
    sdr.setBandwidth(SOAPY_SDR_RX, channel, BW)
    sdr.setBandwidth(SOAPY_SDR_TX, channel, BW)
    constKey = 'TSP_TSG_CONST' if sdr.getDriverKey() == 'iris' else 'TSP_CONST'
    sdr.writeSetting(SOAPY_SDR_TX, channel, constKey, str(1 << 13))

    #per frequency settings
    for freq in FREQS:
        print('Testing freq = %g MHz'%(freq/1e6))
        sdr.setGain(SOAPY_SDR_TX, channel, 'PAD', PAD)
        sdr.setGain(SOAPY_SDR_RX, channel, 'LNA', LNA)
        sdr.setFrequency(SOAPY_SDR_RX, channel, "RF", freq)
        sdr.setFrequency(SOAPY_SDR_TX, channel, "RF", freq+TX_LO_OFS)
        sdr.setFrequency(SOAPY_SDR_RX, channel, 'BB', 0.0)
        sdr.setFrequency(SOAPY_SDR_TX, channel, 'BB', 0.0)

        #calibrate PAD
        if freq in CAL_PAD_LEVELS:
            sdr.setGain(SOAPY_SDR_TX, channel, 'PAD', CAL_PAD_LEVELS[freq])
        else:
            psRx = getRxPowerSpectrum(sdr, rxStream, FFT_SIZE)
            rxLvl = MeasureTonePower(psRx, SAMP_RATE, TX_LO_OFS)
            newPad = PAD+(-20-rxLvl) #-20db tone level
            if newPad > 50: newPad = 50 #clip
            sdr.setGain(SOAPY_SDR_TX, channel, 'PAD', newPad)
            CAL_PAD_LEVELS[freq] = newPad #save for next time
            print('orig level = %g dB, new PAD value = %g dB'%(rxLvl, newPad))

        if calibrate:
            sdr.writeSetting(SOAPY_SDR_RX, channel, 'CALIBRATE', str(BW))
            #tx cal can fail due to weak pad gain for internal loopback
            #try higher gains to get it to pass if it initially fails
            keepGoing = True
            while keepGoing:
                try:
                    sdr.writeSetting(SOAPY_SDR_TX, channel, 'CALIBRATE', str(BW))
                    break
                except:
                    pad = sdr.getGain(SOAPY_SDR_TX, channel, 'PAD') + 5
                    if pad > 50: keepGoing = False
                    print('Failed to cal, trying pad = %g dB'%(pad))
                    sdr.setGain(SOAPY_SDR_TX, channel, 'PAD', pad)
            else:
                raise Exception('Failed to cal tx, gain problems...')
            sdr.setGain(SOAPY_SDR_TX, channel, 'PAD', CAL_PAD_LEVELS[freq]) #restore

        #intentionally disable digital removal to measure the analog removal
        sdr.setDCOffsetMode(SOAPY_SDR_RX, channel, False)

        #data collections
        psRx = getRxPowerSpectrum(sdr, rxStream, FFT_SIZE)
        sdr.setDCOffsetMode(SOAPY_SDR_RX, channel, True) #enable digital dc removal again for the next plot
        if calibrate: time.sleep(RX_DC_SETTLE) #settle digital removal
        sdr.setFrequency(SOAPY_SDR_TX, channel, 'BB', TX_NCO)
        psTx = getRxPowerSpectrum(sdr, rxStream, FFT_SIZE)
        rxLvl = MeasureTonePower(psRx, SAMP_RATE, TX_LO_OFS)
        rxIq = MeasureTonePower(psRx, SAMP_RATE, -TX_LO_OFS)
        rxDc = MeasureTonePower(psRx, SAMP_RATE, 0.0)
        txLvl = MeasureTonePower(psTx, SAMP_RATE, TX_LO_OFS+TX_NCO)
        txIq = MeasureTonePower(psTx, SAMP_RATE, TX_LO_OFS-TX_NCO)
        txDc = MeasureTonePower(psTx, SAMP_RATE, TX_LO_OFS)
        rxDcDig = MeasureTonePower(psTx, SAMP_RATE, 0.0)

        rows.append([freq, psRx, rxLvl, rxIq, rxDc, psTx, txLvl, txIq, txDc, rxDcDig])

        #debug dumps to file
        if dumpDir:
            try: os.makedirs(dumpDir)
            except: pass
            fig = LocalFigure(size=[2, 1], title='Rx and Tx impairments %g MHz, PAD=%g dB'%(freq/1e6, CAL_PAD_LEVELS[freq]))
            fig.addFreqPlot(1, traces=[psRx], rate=SAMP_RATE, title='Tx CW tone into Rx')
            fig.addFreqPlot(2, traces=[psTx], rate=SAMP_RATE, title='Tx w/ NCO into Rx')
            out = os.path.join(dumpDir, 'plot_%04d.png'%int(freq/1e6))
            print(out)
            fig.savefig(out)
            fig.close()

    sdr.closeStream(rxStream)
    sdr = None
    return rows

def main():
    argsStr = sys.argv[1]
    dumpDir = sys.argv[2]

    default_rows = collectSweepData(argsStr, calibrate=False, dumpDir=os.path.join(dumpDir, 'default'))
    calibrated_rows = collectSweepData(argsStr, calibrate=True, dumpDir=os.path.join(dumpDir, 'calibrated'))

    rxLvls = list()
    rxIqsInitial = list()
    rxIqsCorrected = list()
    rxDcsInitial = list()
    rxDcsCorrected = list()
    rxDcsCorrDigital = list()
    txLvls = list()
    txIqsInitial = list()
    txIqsCorrected = list()
    txDcsInitial = list()
    txDcsCorrected = list()

    imagesList = list()
    for default_row, calibrated_row in zip(default_rows, calibrated_rows):
        freq, psRx_DEF, rxLvl_DEF, rxIq_DEF, rxDc_DEF, psTx_DEF, txLvl_DEF, txIq_DEF, txDc_DEF, rxDcDig_DEF = default_row
        freq, psRx_CAL, rxLvl_CAL, rxIq_CAL, rxDc_CAL, psTx_CAL, txLvl_CAL, txIq_CAL, txDc_CAL, rxDcDig_CAL = calibrated_row
        rxLvls.append(rxLvl_CAL)
        rxIqsInitial.append(rxIq_DEF)
        rxIqsCorrected.append(rxIq_CAL)
        rxDcsInitial.append(rxDc_DEF)
        rxDcsCorrected.append(rxDc_CAL)
        rxDcsCorrDigital.append(rxDcDig_CAL)
        txLvls.append(txLvl_CAL)
        txIqsInitial.append(txIq_DEF)
        txIqsCorrected.append(txIq_CAL)
        txDcsInitial.append(txDc_DEF)
        txDcsCorrected.append(txDc_CAL)
        fig = LocalFigure(size=[2, 2], title='Rx and Tx impairments %g MHz, PAD=%g dB'%(freq/1e6, CAL_PAD_LEVELS[freq]))
        fig.addFreqPlot(1, traces=[psRx_DEF], rate=SAMP_RATE, title='Default: Tx CW tone into Rx')
        fig.addFreqPlot(2, traces=[psTx_DEF], rate=SAMP_RATE, title='Default: Tx w/ NCO into Rx')
        fig.addFreqPlot(3, traces=[psRx_CAL], rate=SAMP_RATE, title='Calibrated: Tx CW tone into Rx')
        fig.addFreqPlot(4, traces=[psTx_CAL], rate=SAMP_RATE, title='Calibrated: Tx w/ NCO into Rx')
        out = os.path.join(dumpDir, 'plot_%04d.png'%int(freq/1e6))
        print(out)
        imagesList.append(out)
        fig.savefig(out)
        fig.close()

    fig = LocalFigure(size=[2, 2], title='Summary of impairments for %s'%str(argsStr))

    fig.addDataPlot(1, xaxis=FREQS/1e6, traces=[
        (rxLvls, 'Rx Level'),
        (rxIqsInitial, 'Rx IQ Before'),
        (rxIqsCorrected, 'Rx IQ Corrected'),
    ], title='Rx IQ imbalance calibrations', xlabel='Frequency (MHz)', ylabel='Power (dBfs)', ylim=[-100, 0])

    fig.addDataPlot(2, xaxis=FREQS/1e6, traces=[
        (rxLvls, 'Rx Level'),
        (rxDcsInitial, 'Rx DC Before'),
        (rxDcsCorrected, 'Rx Dc Corrected'),
        (rxDcsCorrDigital, 'Rx Dc w/ TSP'),
    ], title='Rx Dc imbalance calibrations', xlabel='Frequency (MHz)', ylabel='Power (dBfs)', ylim=[-100, 0])

    fig.addDataPlot(3, xaxis=FREQS/1e6, traces=[
        (txLvls, 'Tx Level'),
        (txIqsInitial, 'Tx IQ Before'),
        (txIqsCorrected, 'Tx IQ Corrected'),
    ], title='Tx IQ imbalance calibrations', xlabel='Frequency (MHz)', ylabel='Power (dBfs)', ylim=[-100, 0])

    fig.addDataPlot(4, xaxis=FREQS/1e6, traces=[
        (txLvls, 'Tx Level'),
        (txDcsInitial, 'Tx DC Before'),
        (txDcsCorrected, 'Tx Dc Corrected'),
    ], title='Tx Dc imbalance calibrations', xlabel='Frequency (MHz)', ylabel='Power (dBfs)', ylim=[-100, 0])

    out = os.path.join(dumpDir, 'summary.png')
    print(out)
    imagesList.insert(0, out)
    fig.savefig(out)
    fig.close()

    out = os.path.join(dumpDir, 'summary.pdf')
    print(out)
    os.system('convert %s %s'%(' '.join(imagesList), out))

if __name__ == '__main__':
    main()
