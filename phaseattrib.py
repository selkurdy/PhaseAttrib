"""
PHASEATTRIBGEN.PY.

Each trace of the segy file is read in and a sliding window calculates the phase within the window.
The slpe of the phase is computed and returned for the center sample. The window is shifted by one sample
and repeated.

The resulting segy represents a continuous slope of the phase spectrum over the chosen window.

It was observed that the resulting traces correlates well with gamma ray logs.

Added strict=False for opening segy files for files that do  not have good headers
"""
import os.path
import argparse
import numpy as np
# from scipy.fftpack import fft, ifft
from scipy import stats
import segyio
from shutil import copyfile
from sklearn.preprocessing import MinMaxScaler
# from datetime import datetime
from datetime import datetime


def get_fft_values(amp_values,n_samp, samp_rate=0.002, noise_level=-24):
    """amp_values=amplitudes, samp_rate=sampling rate in seconds, n_samp=number of samples."""
    f_values = np.linspace(0.0, 1.0 / (2.0 * samp_rate), n_samp // 2)
    fft_values_ = np.fft.fft(amp_values)
    # fft_values_norm = 2.0 / n_samp * np.abs(fft_values_[0:n_samp // 2])
    fft_values_org = np.abs(fft_values_[0:n_samp // 2])
    fft_values_db = -20 * np.log10(np.amax(fft_values_org) / fft_values_org)
    fft_phase_ = np.unwrap(np.angle(fft_values_,deg=False))
    fft_phase = fft_phase_[0:n_samp // 2]
    f_values_crop = np.empty([0,1])
    fft_values_db_crop = np.empty([0,1])
    fft_phase_crop = np.empty([0,1])
    for i in range(fft_values_db.size):
        if fft_values_db[i] >= noise_level:
            test0 = f_values[i]
            test1 = fft_values_db[i]
            test2 = fft_phase[i]
            f_values_crop = np.append(f_values_crop,test0)
            fft_values_db_crop = np.append(fft_values_db_crop,test1)
            fft_phase_crop = np.append(fft_phase_crop,test2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(f_values_crop,fft_phase_crop)
    return slope


def centroid_freq(seismic,n_samp, samp_rate=0.002):
    """."""
    f_values = np.linspace(0.0, 1.0 / (2.0 * samp_rate), n_samp // 2)
    fft_values = np.fft.fft(seismic)
    prspec = np.abs(fft_values[0:n_samp // 2])**2
    specsum = prspec[0]
    frsp = f_values[0] * prspec[0]
    for j in range(1,f_values.size):
        specsum = specsum + prspec[j]
        frsp = frsp + (f_values[j] * prspec[j])
    fs = frsp / specsum
    return fs

def seis_ph_slp_cf(seismic,phase_slope,scaleopt,wl=256):
    """."""
    t = np.linspace(0,int(wl - 2),int(wl / 2))
    if scaleopt[0]:
        trsclmm = MinMaxScaler((scaleopt[1],scaleopt[2]))

    copyfile(seismic,phase_slope)
    start_process = datetime.now()
    with segyio.open(seismic, "r",strict=False) as seis:
        with segyio.open(phase_slope,"r+",strict=False)as phslp:
            trnum = 0
            for seistr,phslptr in zip(seis.trace,phslp.trace):
                if not np.all(seistr == 0):
                    for i in range(int(wl / 4),(seistr.size - int(wl / 4))):
                            winseistr = seistr[i - int(wl / 4):i + int(wl / 4)]
                            fs = centroid_freq(winseistr,int(wl / 2))
                            if fs != 0:
                                sig = 4.5933 * pow(fs,-0.959)
                                gs = np.exp(-0.5 * pow(((t - int(wl / 2)) / (sig * int(wl / 2))),2))
                                hldr = gs * winseistr
                                try:
                                    slp=get_fft_values(hldr,int(wl/2))
                                except:
                                    slp=0
                                phslptr[i] = slp
                            else:
                                phslptr[i] = 0
                phslptr[0:int(wl / 4)] = 0
                phslptr[phslptr.size - (int(wl / 4) + 1):phslptr.size] = 0
                if scaleopt[0]:
                    phslptr = trsclmm.fit_transform(phslptr.reshape(-1,1))
                    phslptr = phslptr.reshape(-1)
                phslp.trace[trnum] = phslptr
                if trnum % 1000 == 0:
                    print(f'Processing Trace#: {trnum:10d}')
                trnum += 1
    end_process = datetime.now()
    print('Duration of processing: {}'.format(end_process - start_process))

def dom_freq_ct(seismic):
    """Compute dominant frequency using ."""
    grd = np.gradient(seismic)
    # pk = np.array([])
    pk = []
    for i in range(grd.size - 1):
        if (((grd[i] == 0) or (grd[i] > 0)) and (grd[i + 1] < 0)) and (seismic[i] > 0):
            pk = np.append(pk,grd[i])
    # domfreq = (pk.size / (seismic.size * 2)) * 1000
    domfreq = (len(pk) / (seismic.size * 2)) * 1000
    return domfreq

def seis_ph_slp_dfct(seismic,phase_slope,scaleopt,wl=256):
    """."""
    t = np.linspace(0,int(wl - 2),int(wl / 2))
    if scaleopt[0]:
        trsclmm = MinMaxScaler((scaleopt[1],scaleopt[2]))
    copyfile(seismic,phase_slope)
    start_process = datetime.now()
    with segyio.open(seismic, "r",strict=False) as seis:
        with segyio.open(phase_slope,"r+",strict=False)as phslp:
            trnum = 0
            for seistr,phslptr in zip(seis.trace,phslp.trace):
                if not np.all(seistr == 0):
                    for i in range(int(wl / 4),(seistr.size - int(wl / 4))):
                            winseistr = seistr[i - int(wl / 4):i + int(wl / 4)]
                            domfreq = dom_freq_ct(winseistr)
                            if domfreq != 0:
                                sig = 4.5933 * pow(domfreq,-0.959)
                                gs = np.exp(-0.5 * pow(((t - int(wl / 2)) / (sig * int(wl / 2))),2))
                                hldr = gs * winseistr
                                try:
                                    slp=get_fft_values(hldr,int(wl/2))
                                except:
                                    slp=0
                                phslptr[i] = slp
                            else:
                                phslptr[i] = 0
                phslptr[0:int(wl / 4)] = 0
                phslptr[phslptr.size - (int(wl / 4) + 1):phslptr.size] = 0
                if scaleopt[0]:
                    phslptr = trsclmm.fit_transform(phslptr.reshape(-1,1))
                    phslptr = phslptr.reshape(-1)
                phslp.trace[trnum] = phslptr
                if trnum % 1000 == 0:
                    print(f'Processing Trace#: {trnum:10d}')
                trnum += 1
    end_process = datetime.now()
    print('Duration of processing: {}'.format(end_process - start_process))

def dom_freq_ft(amp_values,n_samp, samp_rate=0.002):
    """."""
    f_values = np.linspace(0.0, 1.0 / (2.0 * samp_rate), n_samp // 2)
    fft_values = np.fft.fft(amp_values)
    prspec = np.abs(fft_values[0:n_samp // 2])**2
    domfreq = f_values[np.where(prspec == max(prspec))]
    return domfreq

def seis_ph_slp_dfft(seismic,phase_slope,scaleopt,wl=256):
    """."""
    t = np.linspace(0,int(wl - 2),int(wl / 2))
    if scaleopt[0]:
        trsclmm = MinMaxScaler((scaleopt[1],scaleopt[2]))
    copyfile(seismic,phase_slope)
    start_process = datetime.now()
    with segyio.open(seismic, "r",strict=False) as seis:
        with segyio.open(phase_slope,"r+",strict=False)as phslp:
            trnum = 0
            for seistr,phslptr in zip(seis.trace,phslp.trace):
                if not np.all(seistr == 0):
                    for i in range(int(wl / 4),(seistr.size - int(wl / 4))):
                            winseistr = seistr[i - int(wl / 4):i + int(wl / 4)]
                            domfreq = dom_freq_ft(winseistr,int(wl / 2))
                            if domfreq != 0:
                                sig = 4.5933 * pow(domfreq,-0.959)
                                gs = np.exp(-0.5 * pow(((t - int(wl / 2)) / (sig * int(wl / 2))),2))
                                hldr = gs * winseistr
                                try:
                                    slp=get_fft_values(hldr,int(wl/2))
                                except:
                                    slp=0
                                phslptr[i] = slp
                            else:
                                phslptr[i] = 0
                phslptr[0:int(wl / 4)] = 0
                phslptr[phslptr.size - (int(wl / 4) + 1):phslptr.size] = 0
                if scaleopt[0]:
                    phslptr = trsclmm.fit_transform(phslptr.reshape(-1,1))
                    phslptr = phslptr.reshape(-1)
                phslp.trace[trnum] = phslptr
                if trnum % 1000 == 0:
                    print(f'Processing Trace#: {trnum:10d}')
                trnum += 1
    end_process = datetime.now()
    print('Duration of processing: {}'.format(end_process - start_process))


def getcommandline():
    """."""
    parser = argparse.ArgumentParser(description='Process Phase Slope')
    parser.add_argument('ampsegyfilename',help='Amplitude segy file name')
    parser.add_argument('--winlength',type=int,default=256,
        help='Window length in ms.Has to be power of 2.default=256')
    parser.add_argument('--domfreqmethod',choices=['pkcount','centroidfreq','peakfreq'],default='pkcount',
        help='Dominant fequency calculation method: pkcount=counting,centroidfreq=centoid frequency,peakfreq=from fft. default=cnt')
    parser.add_argument('--scale2gr',nargs=3,type=int,default=(1,0,150),
        help='Min Max values to scale output computed Psuedo GR trace. default 1 0 150')
    # do not scale : use --scale2gr 0 0 150. The first zero is a boolean
    # 0 is false 1 is true, min then max gr values to scale to

    result = parser.parse_args()
    return result


def main():
    """."""
    cmdl = getcommandline()
    dirsplit,fextsplit = os.path.split(cmdl.ampsegyfilename)
    fname,fext = os.path.splitext(fextsplit)
    phaseslopefname = fname + '_phslp_%s_%d.sgy' % (cmdl.domfreqmethod,cmdl.winlength)
    if cmdl.domfreqmethod == 'centroidfreq':
        seis_ph_slp_cf(cmdl.ampsegyfilename,phaseslopefname,cmdl.scale2gr,wl=cmdl.winlength)
    elif cmdl.domfreqmethod == 'pkcount':
        seis_ph_slp_dfct(cmdl.ampsegyfilename,phaseslopefname,cmdl.scale2gr,wl=cmdl.winlength)
    elif cmdl.domfreqmethod == 'peakfreq':
        seis_ph_slp_dfft(cmdl.ampsegyfilename,phaseslopefname,cmdl.scale2gr,wl=cmdl.winlength)

if __name__ == '__main__':
    main()
