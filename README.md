# cTRF

The cTRF toolbox relates a feature of a natural stimulus, such as speech, to the neural response through computing *complex temporal response functions (cTRFs)*. This complex modeling can identify both the temporal lag as well as phase of the neural response. It has been employed successfully to relate the pitch waveform of speech to the human brainstem response as measured from scalp electrodes, and to thereform  decode selective attention to one of two speakers [1,2].


**Content**

* cTRF.py - a set of custom functions for cTRF modelling

* demo.ipynb - an example 

* FW_short.npy - sample of the fundamental waveform of a speech signal (short sample only)

* eeg_short_raw.fif - sample of the EEG recording that corresponds to the speech stimulus  (a short sample only)

**Required packages**

MNE https://martinos.org/mne/dev/index.html
NumPy http://www.numpy.org/
SciPy https://www.scipy.org/
Matplotlib https://matplotlib.org/


**Credit**

Mikolaj Kegler, Octave Etard, Antonio Elia Forte, Tobias Reichenbach (Imperial College London)


**References**

1. A. E. Forte, O. Etard and T. Reichenbach, The human auditory brainstem response to running speech reveals a subcortical mechanism for selective attention, eLife 6:e27203 (2017).

2. O. Etard, M. Kegler, C. Braiman, A. E. Forte, T. Reichenbach,
Real-time decoding of selective attention from the human auditory brainstem response to continuous speech, https://www.biorxiv.org/content/early/2018/02/05/259853

