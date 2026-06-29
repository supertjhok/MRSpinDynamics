% Notes on OCT excitation pulses for RP2-1.0 refocusing and T_E = 7 x T_180
% Segment length is in units of T_180

load dat_files\results_mag_all.mat

%Name, index number, segment length, total length, normalized SNR (power units)
%-----------------------------------------------------------------------------
OCT,  74,    0.1000,   10.0000,    2.7389
OCT_A, 1,    0.0780,   12.6360,    3.2119
OCT_B, 2,    0.0780,   15.7560,    3.1904
OCT_C, 4,    0.0780,   12.1680,    3.1106
OCT_D, 9,    0.0780,   11.2320,    3.0623
OCT_E, 33,   0.0780,   10.2960,    2.9165
OCT_F, 47,   0.0780,   8.5800,    2.8516

% Notes on OCT excitation pulses for Rectangular refocusing and T_E = 7 x T_180
% Segment length is in units of T_180

% Refocusing pulse lengths decrease from 0.99*pi in 0.01*pi steps
load dat_files\results_mag14.mat

% Only the following refocusing pulses:
% pi (180 degrees), 3*pi/4 (135 degrees), and 0.69*pi (124 degrees) (best performance)
load dat_files\OCT_rect_pulses.mat 

% Notes on OCT excitation pulses for antisymmetric 0/pi refocusing pulses,
% n x T_180, T_acq = 5 x T_180 
% ----------------------------------------------------------------------------------

Case 1
% ----------------------------------------------------------------------------------
Refocusing pulse:
results_ref_mag1_10_run4.mat, pulse number = 1
segment length = 0.1 x T_180, number of segments = 20, total length = 2 x T_180
Bruker implementation = OCT_REF_2X_1

Excitation pulse: 
results_mag16_1_run1.mat, pulse number = 5
segment length = 0.08 x T_180, number of segments = 180, total length = 14.4 x T_180
normalized SNR (power units) =  4.4531 (after phase inversion cycle)
Bruker implementation = OCT_EXC_180_A_1 (and)  OCT_EXC_180_A_2 (PI pair)

Case 2
% ----------------------------------------------------------------------------------
Refocusing pulse:
results_ref_mag2_11_run1, pulse number = 19
segment length = 0.1 x T_180, number of segments = 21, total length = 2.1 x T_180
Bruker implementation = OCT_REF_2X_2

Excitation pulse:
results_mag16_19_run2.mat, pulse number = 4
segment length = 0.08 x T_180, number of segments = 180, total length = 14.4 x T_180
normalized SNR (power units) = 4.6455 (after phase inversion cycle)
Bruker implementation = OCT_EXC_180_B_1 (and)  OCT_EXC_180_B_2 (PI pair)

Case 3
% ----------------------------------------------------------------------------------
Refocusing pulse:
results_ref_mag4_11_run1.mat, pulse number = 11
segment length = 0.01 x T_180, number of segments = 213, total length = 2.13 x T_180
Bruker implementation = OCT_REF_2X_3

Excitation pulse:
results_mag16_11_run2.mat, pulse number = 4
segment length = 0.08 x T_180, number of segments = 180, total length = 14.4 x T_180
normalized SNR (power units) = 4.5009 (after phase inversion cycle)
Bruker implementation = OCT_EXC_180_C_1 (and)  OCT_EXC_180_C_2 (PI pair)

Case 4
% ----------------------------------------------------------------------------------
Refocusing pulse:
results_ref_mag2_11_run1, pulse number = 19
segment length = 0.1 x T_180, number of segments = 21, total length = 2.1 x T_180
Bruker implementation = OCT_REF_2X_2

Excitation pulse:
results_mag16_19_run4.mat, pulse number = 3
segment length = 0.08 x T_180, number of segments = 180, total length = 14.4 x T_180
normalized SNR (power units) =  4.7722 (after phase inversion cycle)
Bruker implementation = OCT_EXC_180_D_1 (and)  OCT_EXC_180_D_2 (PI pair)

Case 5
% ----------------------------------------------------------------------------------
Refocusing pulse:
results_ref_mag2_16_run1.mat, pulse number = 16
segment length = 0.1 x T_180, number of segments = 31, total length = 3.1 x T_180
Bruker implementation = OCT_REF_3X_1

Excitation pulse:
results_mag16_16_run7.mat, pulse number = 3
segment length = 0.08 x T_180, number of segments = 180, total length = 14.4 x T_180
normalized SNR (power units) = 5.0460 (after phase inversion cycle)
Bruker implementation = OCT_EXC_180_E_1 (and)  OCT_EXC_180_E_2 (PI pair)

Case 6
% ----------------------------------------------------------------------------------
Refocusing pulse:
results_ref_mag2_16_run1.mat, pulse number = 16
segment length = 0.1 x T_180, number of segments = 31, total length = 3.1 x T_180
Bruker implementation = OCT_REF_3X_1

Excitation pulse:
results_mag16_16_run10.mat, pulse number = 5
segment length = 0.05 x T_180, number of segments = 288, total length = 14.4 x T_180
normalized SNR (power units) = 5.7061 (after phase inversion cycle)
Bruker implementation = OCT_EXC_180_F_1 (and)  OCT_EXC_180_F_2 (PI pair)
