# Longitudinal changes in aperiodic and periodic activity in electrophysiological recordings in the first seven months of life

This repository provides analysis code to analyze longitudinal changes in aperiodic activity in infant EEG data. The repository code recreates results and figures from the following manuscript:

# Reference
Schaworonkow N & Voytek B: [Longitudinal changes in aperiodic and periodic activity in electrophysiological recordings in the first seven months of life.](https://www.sciencedirect.com/science/article/pii/S1878929320301420) _Developmental Cognitive Science_ (2021). doi:10.1016/j.dcn.2020.100895

# Dataset
The results are based on following available openly available data set: [infant EEG dataset](https://figshare.com/articles/infant_EEG_data/5598814) and the corresponding [data sheet](https://figshare.com/articles/Relative_Power_EEG_and_Bayley_Scales_Infant_data/6994946).

From the associated articles:
-  Xiao R, Shida-Tokeshi J ,Vanderbilt DL, Smith BA: [Electroencephalography power and coherence changes with age and motor skill development across the first half year of life.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190276) _PLOSOne_ (2018).
- Hooyman A, Kayekjian D, Xiao R, Jiang C, Vanderbilt DL, Smith BA: [Relationships between variance in electroencephalography relative power and developmental status in infants with typical development and at risk for developmental disability: An observational study.](https://gatesopenresearch.org/articles/2-47/v2) _GatesOpenResearch_ (2018).

To reproduce the results, the data set should be downloaded and placed in the folder ```data```.

# Requirements

The provided python3 scripts are using ```scipy``` and ```numpy``` for general computation, ```pandas``` for saving intermediate results to csv-files. ```matplotlib``` for visualization. For EEG-related analysis, the ```mne``` package is used. For computation of aperiodic exponents: [```fooof```](https://fooof-tools.github.io/fooof/) and for computation of waveform features: [```bycycle```](https://bycycle-tools.github.io/bycycle/). Specifically used versions can be seen in the ```requirements.txt```. R-scripts use [```lme4```](https://cran.r-project.org/web/packages/lme4/index.html) and [```ciTools```](https://cran.r-project.org/web/packages/ciTools/index.html).


# Pipeline

To reproduce the figures from the command line, navigate into the ```code``` folder and execute ```make all```. This will run through the preprocessing steps, the analysis of aperiodic exponents and the oscillatory burst analysis. The scripts can also be executed separately in the order described in the ```Makefile```.
