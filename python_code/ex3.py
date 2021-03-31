import numpy as np                      #for arrays and matrices
import matplotlib.pyplot as plt         #for ploting
import scipy                            #for math calculations
from mpl_toolkits.mplot3d import Axes3D #for 3D ploting
import math                             #for math constants
from matplotlib import collections as matcoll
from scipy import signal                #for signal analysis
from scipy import fftpack               #for fourier spectrum
from scipy.fftpack import fft

##PARAMETERS
Fm = 3000 #kHz
Tm = 1 / Fm #sec
A = 1 #V
AM = 3 
N_periods = 4 #periods displayed

Samples_per_period = 2000 #number of samples per period
N_samples = N_periods * Samples_per_period + 1 #total number of samples (in linspace)
Timestep = 1.0 / (float(Fm * Samples_per_period)) #sample spacing

#### Α ΕΡΏΤΗΜΑ ####
A_bit = Fm/1000 #(V) Amplitude of bit stream
T_b = 0.5 #(sec) bit duration 
N_rand_bits = 46 #number of random bits generated

rand_bits = np.random.randint(2, size=(N_rand_bits)) #generate random bits [0,1]

samples_per_bit = 100
rand_bits_linspace = []
for i in range(0, len(rand_bits)):  
    for j in range(0, samples_per_bit):
        rand_bits_linspace.append(rand_bits[i])

t_rand_bits = np.linspace(0, T_b*N_rand_bits, N_rand_bits*samples_per_bit, endpoint=False)
y_rand_bits = A_bit*signal.square(2*math.pi*t_rand_bits, duty=rand_bits_linspace[0:N_rand_bits*samples_per_bit])
plt.plot(t_rand_bits, y_rand_bits, label='B-PAM')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.title('B-PAM modulation of random bits')   
plt.xlabel('Time (sec)'); plt.ylabel('Amplitude (V)')
plt.legend(loc='upper left')
# plt.show()
plt.figure()

#### B ΕΡΏΤΗΜΑ ####
E_b = pow(A_bit, 2)*T_b 

x_bpam = [-math.sqrt(E_b), math.sqrt(E_b)]
y_bpam = [0, 0]
plt.scatter(x_bpam,y_bpam)
plt.ylim([-0.5, 0.5])
plt.grid(True, which='both')
plt.title('Constellation of B-PAM')   
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
# plt.show()
plt.figure()

#### Γ ΕΡΏΤΗΜΑ ####
#SNR to linear scale function
def SNR_dB_lin(snr_ratio):
    return 10**(snr_ratio / 10)

No_5 = E_b / SNR_dB_lin(5) #Conversion from dB to linear scale
No_15 = E_b / SNR_dB_lin(15) #Conversion from dB to linear scale

plt.subplots_adjust(hspace=0.5)
#Eb/No (SNR) = 5 dB
# awgn_5 = np.sqrt(No_5/2)*np.random.standard_normal(N_rand_bits*samples_per_bit)
awgn_5 = np.random.normal(0, np.sqrt(No_5), 2*N_rand_bits*samples_per_bit).view(np.complex128) #complex awgn (5dB)
# awgn_5_im = np.random.normal(0, math.sqrt(No_5), N_rand_bits*samples_per_bit) #imaginary part of awgn (5dB)
plt.subplot(3, 1, 2)
plt.plot(t_rand_bits, y_rand_bits + awgn_5.real, label='Eb/No = 5')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.title('Eb/No=5')   
plt.ylabel('Amplitude (V)')
plt.legend(loc='upper left')

#Eb/No (SNR) = 15 dB
# awgn_15 = np.sqrt(No_15/2)*np.random.standard_normal(N_rand_bits*samples_per_bit)
awgn_15 = np.random.normal(0, np.sqrt(No_15), 2*N_rand_bits*samples_per_bit).view(np.complex128) #complex awgn (15dB)
# awgn_15_im = np.random.normal(0, math.sqrt(No_15), N_rand_bits*samples_per_bit) #imaginary part of awgn (15dB)
plt.subplot(3, 1, 3)
plt.plot(t_rand_bits, y_rand_bits + awgn_15.real, label='Eb/No = 15')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.title('Eb/No=15')   
plt.xlabel('Time (sec)'); plt.ylabel('Amplitude (V)')
plt.legend(loc='upper left')

#Initial B-PAM signal
plt.subplot(3, 1, 1)
plt.plot(t_rand_bits, y_rand_bits, label='B-PAM')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.title('B-PAM modulation of random bits')   
plt.ylabel('Amplitude (V)')
plt.legend(loc='upper left')
plt.show()
# plt.figure()

#### Δ ΕΡΏΤΗΜΑ ####
#Eb/No = 5 dB
x_bpam_awgn_5 = (y_rand_bits[::1] + awgn_5.real[::1]) * math.sqrt(T_b)
y_bpam_awgn_5 = (awgn_5.imag[::1]) * math.sqrt(T_b)
# y_bpam_awgn_5 = np.zeros(N_rand_bits*samples_per_bit // 50)

plt.subplot(2, 1, 1)
plt.scatter(x_bpam_awgn_5 ,y_bpam_awgn_5, s=2.5, c='b', label='Eb/No = 5')
plt.ylim([-0.5, 0.5])
plt.grid(True, which='both')
plt.title('Constellation of B-PAM')   
plt.legend(loc='upper left')
plt.ylabel('Quadrature')

#Eb/No = 15 dB
x_bpam_awgn_15 = (y_rand_bits[::1] + awgn_15.real[::1]) * math.sqrt(T_b)
y_bpam_awgn_15 = (awgn_15.imag[::1]) * math.sqrt(T_b)
# y_bpam_awgn_15 = np.zeros(N_rand_bits*samples_per_bit // 50)

plt.subplot(2, 1, 2)
plt.scatter(x_bpam_awgn_15 ,y_bpam_awgn_15, s=2, c='b', label='Eb/No = 15')
plt.ylim([-0.5, 0.5])
plt.grid(True, which='both') 
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
plt.legend(loc='upper left')
plt.show()

#### Ε ΕΡΏΤΗΜΑ ####
N_rand_bits_2 = 10**6
rand_bits_2 = np.random.randint(2, size=(N_rand_bits_2)) #random bits generated
rand_bits_2_mod = rand_bits_2*2*A_bit-A_bit #rand bits modulated (1->A_bit, 0->-A_bit)
t_BER = np.arange(0, 16) #linear space

#EXPERIMENTAL
No_exp, awgn_exp, BER_exp = [], [], []
for i in t_BER:
    No_exp.append(E_b / SNR_dB_lin(i))
    awgn_exp.append(np.random.normal(0, np.sqrt(No_exp[i]), 2*N_rand_bits_2).view(np.complex128))
    output_sign = rand_bits_2_mod + awgn_exp[i].real
    receiv_sign = (output_sign >= 0).astype(int)
    BER_exp.append(np.sum(receiv_sign != rand_bits_2) / N_rand_bits_2)

#THEORETICAL
# BER_theor = scipy.special.erfc(np.sqrt(SNR_dB_lin(t_BER)))
def q_bpam(a):
    return (1.0/math.sqrt(2*math.pi))*scipy.integrate.quad(lambda x: math.exp(-(x**2)/2), a, pow(10,2))[0]
BER_theor = []
for i in t_BER:
    BER_theor.append(q_bpam(np.sqrt(2*SNR_dB_lin(i))))

plt.semilogy(t_BER, BER_exp, color='r', marker='o', markersize=2, linestyle='')
plt.semilogy(t_BER, BER_theor, marker='', linestyle='-', linewidth=1 )
plt.title('Experimental & theoretical BER curve')
plt.xlabel('$E_b/N_0(dB)$');plt.ylabel('BER ($P_e$)')
plt.grid(True)
plt.show()


# t_ber_theor = np.linspace(0, 15, 150, endpoint=False)
# y_ber_theor = []
# y_ber_theor = q_bpam(np.sqrt(2*pow(10, t_ber_theor/10)))
# plt.semilogy(t_ber_theor, y_ber_theor)
# plt.show()
