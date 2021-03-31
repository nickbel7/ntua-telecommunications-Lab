import numpy as np                      #for arrays and matrices
import matplotlib.pyplot as plt         #for ploting
import scipy                            #for math calculations
from mpl_toolkits.mplot3d import Axes3D #for 3D ploting
import math                             #for math constants
from matplotlib import collections as matcoll
from scipy import signal                #for signal analysis
from scipy import fftpack               #for fourier spectrum
from scipy.fftpack import fft
import binascii
from scipy.io import wavfile

'''
©Editors : 
Nick Bellos
Magdalini Efthymiadou
'''

########### QUESTION 1 #########

##PARAMETERS
Fm = 3000 #kHz
Tm = 1 / Fm #sec
A = 1 #V
AM = 3 
N_periods = 4 #periods displayed

Samples_per_period = 2000 #number of samples per period
N_samples = N_periods * Samples_per_period + 1 #total number of samples (in linspace)
Timestep = 1.0 / (float(Fm * Samples_per_period)) #sample spacing

#INITIAL SIGNAL
t = np.linspace(0, N_periods*Tm, N_samples)
y = A*np.cos(2*math.pi*Fm*t)*np.cos(2*math.pi*(AM+2)*Fm*t)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) #for automatic x-scale (10^-3)
plt.plot(t, y, '-', marker="", markersize=4, label="y(t)") #for markers
plt.title('Initial Signal y(t)')
plt.xlabel('Time (s)'); plt.ylabel('Amplitude (Volt)') 
plt.legend(loc='upper left')
##plt.savefig('ex1-photos/Initial-Signal-y(t).png')
plt.show()

#### Α ΕΡΏΤΗΜΑ ####
def sign_sampling(coeff):
    t_freq = np.linspace(0, N_periods*Tm, N_periods*coeff+1)
    y_freq = A*np.cos(2*math.pi*Fm*t_freq)*np.cos(2*math.pi*(AM+2)*Fm*t_freq)
    plt.vlines(t_freq, [0], y_freq, linewidth=0.3)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) #for automatic x-scale (10^-3)
    plt.plot(t_freq, y_freq, '.', marker=".", markersize=4, label='fs='+ str(coeff) +'fm')
    plt.title('Sampling with fs = '+ str(coeff) +'fm')
    plt.xlabel('Time (s)'); plt.ylabel('Amplitude (Volt)')  
    plt.legend(loc='upper left')
    ##plt.savefig('ex1-photos/'+str(coeff)+'fm-sampling.png')
    plt.show()

#Ι) -> fs1=20fm
sign_sampling(20)

#ΙΙ) -> fs2=100fm 
sign_sampling(100)

#III) -> fs1=20fm & fs2=100fm 
t_20 = np.linspace(0, N_periods*Tm, 4*20+1)
y_20 = A*np.cos(2*math.pi*Fm*t_20)*np.cos(2*math.pi*(AM+2)*Fm*t_20)
plt.vlines(t_20, [0], y_20, linewidth=0.8, colors="b")

t_100 = np.linspace(0, N_periods/Fm, 4*100+1)
y_100 = A*np.cos(2*math.pi*Fm*t_100)*np.cos(2*math.pi*(AM+2)*Fm*t_100)
plt.vlines(t_100, [0], y_100, linewidth=0.3)

plt.plot(t_20, y_20, '.', t_100, y_100, '.')
plt.title('Sampling with fs1=20fm & fs2=100fm')
plt.xlabel('Time (s)'); plt.ylabel('Amplitude (Volt)')  
plt.legend(['fs1 = 20fm', 'fs2 = 100fm'], loc='upper left')
##plt.savefig('ex1-photos/20fm_100fm-sampling.png')
plt.show()

#### B ΕΡΏΤΗΜΑ ####
sign_sampling(5)

#FOURIER SPECTRUM
plt.subplot(2, 1, 1)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) #for automatic x-scale (10^-3)
plt.plot(t, y, label='y(t)')
plt.title('Fourier spectrum of y(t)')
plt.xlabel('Time (s)'); plt.ylabel('Amplitude (Volt)') 
plt.legend(loc='upper left')
y_fft = fft(y)
t_fft = fftpack.fftfreq(N_samples, Timestep)
t_fft = fftpack.fftshift(t_fft)
y_fft_plot = fftpack.fftshift(y_fft)
plt.subplot(2, 1, 2)
plt.xlim(0, 8*Fm)
plt.xticks(np.arange(0, 11*Fm, step=Fm))
plt.plot(t_fft, float(N_periods/N_samples) * np.abs(y_fft_plot), label='Fourier spectrum')
plt.xlabel('Frequency (Hz)'); plt.ylabel('Amplitude (Volt)')
plt.grid()
plt.legend(loc='upper left')
##plt.savefig('ex1-photos/Fourier-spectrum-y(t).png')
plt.show()


########### QUESTION 2 #########

##PARAMETERS
if (Fm/1000 % 2):               #bits for quantization
    bits = 5 
else: 
    bits = 4                    
q_levels = 2**bits              #quantization levels
q_levels_top = q_levels/2       #quantization levels on one side             
s_max = max(abs(y_20))          #get the max value
delta = (2*s_max)/(q_levels-1)  #step size

#### Α ΕΡΏΤΗΜΑ ####
#QUANTIZATION
quant_signal = np.copy(y_20) #np.copy() copies y_20 array without reference
y_20_new = np.copy(y_20)
for i in range(0,y_20.size):
    quant_signal[i] = int(math.floor(round(y_20[i],4)/delta)) #quantized levels (int)|(Min: -2**bits/2, Max: 2**bits/2-1)
    y_20_new[i] = delta*(quant_signal[i])+delta/2 #mid-riser quantized signal

#GRAY CODE GENERATOR
def gray_code(n_bits):
    gray_arr = list()
    gray_arr.append("0")
    gray_arr.append("1")
    i = 2
    j = 0
    while(True):
        if i>=1 << n_bits:
            break
        for j in range(i - 1, -1, -1):
            gray_arr.append(gray_arr[j])
        for j in range(i):
            gray_arr[j] = "0" + gray_arr[j]
        for j in range(i, 2 * i):
            gray_arr[j] = "1" + gray_arr[j]
        i = i << 1

    return gray_arr

#PLOT FOR QUANTIZED SIGNAL
gray_code_ex2 = gray_code(bits)
plt.vlines(t_20, [0], quant_signal, linewidth=0.8, colors="b")
plt.yticks(np.arange(-q_levels/2, q_levels/2, 1), gray_code_ex2)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) #for automatic x-scale (10^-3)
plt.plot(t_20, quant_signal, '.', label='fs1=20fm')
plt.title('y(t) quantized with mid riser')   
plt.xlabel('Time (sec)'); plt.ylabel('Gray Code')
plt.legend(loc='upper left')
##plt.savefig('ex2-photos/y(t)-mid-riser.png')
plt.show()


#### Β ΕΡΏΤΗΜΑ ####
#Ι) -> variance for 10 samples
error_10 = y_20[0:10]-y_20_new[0:10]
var_10 = (1/10)*sum(map(lambda x:x*x,error_10))
print('Variance for first \n10 samples : '+ str(var_10))

#ΙI) -> variance for 20 samples
error_20 = y_20[0:20]-y_20_new[0:20]
var_20 = (1/20)*sum(map(lambda x:x*x,error_20))
print('20 samples : '+ str(var_20))
        
#III) -> SNR
error_quant = (1/3)*pow(A,2)*pow(2,-2*bits) #Τυπική απόκλιση σφάλματος κβάντισης
P_mean_20 = (1/20)*sum(map(lambda x:x*x,y_20[0:20])) #Mέση ισχύς του σήματος y_20(t)
SNR_10 = P_mean_20 / var_10 #SNR for 10 samples
print('SNR for 10 samples : '+ str(SNR_10))
SNR_20 = P_mean_20 / var_20 #SNR for 20 samples
print('SNR for 20 samples : '+ str(SNR_20))
SNR_theor = P_mean_20 / error_quant #SNR for theoretical value
print('SNR for theoretical : '+ str(SNR_theor))


#### Γ ΕΡΏΤΗΜΑ ####
#creates string of bitstream
Bitstream = '' #bit stream of output (string)
for i in range(0, 20):
    Bitstream += gray_code_ex2[int(quant_signal[i]+q_levels/2)] 

#appends bits to array for ploting
samples_per_bit = 100
polar_nrz = []
for i in range(0, len(Bitstream)):
    for j in range(0, samples_per_bit):
        polar_nrz.append(int(Bitstream[i])) 

t_bit_20 = np.linspace(0, 0.001*bits*20, samples_per_bit*bits*20, endpoint=False)
plt.plot(t_bit_20, Fm/1000*signal.square(2*math.pi*t_bit_20, duty=polar_nrz[0:samples_per_bit*bits*20]), label='POLAR NRZ')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.title('Bit stream of quantized signal (fs1=20fm)')   
plt.xlabel('Time (sec)'); plt.ylabel('Amplitude (V)')
plt.legend(loc='upper left')
##plt.savefig('ex2-photos/y(t)-bit-stream.png')
plt.show()


########### QUESTION 3 #########

##PARAMETERS
A_bit = Fm/1000 #(V) Amplitude of bit stream
T_b = 0.5 #(sec) bit duration 
N_rand_bits = 46 #number of random bits generated

#### Α ΕΡΏΤΗΜΑ ####
rand_bits = np.random.randint(2, size=(N_rand_bits)) #generate random bits [0,1]

#Β-PAM modulation
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
##plt.savefig('ex3-photos/B-PAM-modulation.png')
plt.show()

#### B ΕΡΏΤΗΜΑ ####
#Constellation diagram for B-PAM signal (without noise)
E_b = pow(A_bit, 2)*T_b 

x_bpam = [-math.sqrt(E_b), math.sqrt(E_b)]
y_bpam = [0, 0]
plt.scatter(x_bpam,y_bpam)
plt.ylim([-0.5, 0.5])
plt.grid(True, which='both')
plt.title('Constellation of B-PAM')   
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
##plt.savefig('ex3-photos/B-PAM-constellation.png')
plt.show()

#### Γ ΕΡΏΤΗΜΑ ####
#SNR to linear scale function
def SNR_dB_lin(snr_ratio):
    return 10**(snr_ratio / 10)

No_5 = E_b / SNR_dB_lin(5) #Conversion from dB to linear scale
No_15 = E_b / SNR_dB_lin(15) #Conversion from dB to linear scale

plt.subplots_adjust(hspace=0.5)
#Eb/No (SNR) = 5 dB
awgn_5 = np.random.normal(0, np.sqrt(No_5), 2*N_rand_bits*samples_per_bit).view(np.complex128) #complex awgn (5dB)
plt.subplot(3, 1, 2)
plt.plot(t_rand_bits, y_rand_bits + awgn_5.real, label='Eb/No = 5')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.title('Eb/No=5')   
plt.ylabel('Amplitude (V)')
plt.legend(loc='upper left')

#Eb/No (SNR) = 15 dB
awgn_15 = np.random.normal(0, np.sqrt(No_15), 2*N_rand_bits*samples_per_bit).view(np.complex128) #complex awgn (15dB)
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
##plt.savefig('ex3-photos/AWGN-5dB-15dB.png')
plt.show()

#### Δ ΕΡΏΤΗΜΑ ####
#Constellation diagram for Eb/No = 5 dB
x_bpam_awgn_5 = (y_rand_bits[::1] + awgn_5.real[::1]) * math.sqrt(T_b)
y_bpam_awgn_5 = (awgn_5.imag[::1]) * math.sqrt(T_b)

plt.subplot(2, 1, 1)
plt.scatter(x_bpam_awgn_5 ,y_bpam_awgn_5, s=1, c='b', label='Eb/No = 5')
plt.ylim([-0.5, 0.5])
plt.grid(True, which='both')
plt.title('Constellation of B-PAM')   
plt.legend(loc='upper left')
plt.ylabel('Quadrature')

#Constellation diagram for Eb/No = 15 dB
x_bpam_awgn_15 = (y_rand_bits[::1] + awgn_15.real[::1]) * math.sqrt(T_b)
y_bpam_awgn_15 = (awgn_15.imag[::1]) * math.sqrt(T_b)

plt.subplot(2, 1, 2)
plt.scatter(x_bpam_awgn_15 ,y_bpam_awgn_15, s=1, c='b', label='Eb/No = 15')
plt.ylim([-0.5, 0.5])
plt.grid(True, which='both') 
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
plt.legend(loc='upper left')
##plt.savefig('ex3-photos/AWGN-constellation.png')
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
    awgn_exp.append(np.random.normal(0, np.sqrt(No_exp[i]), 2*N_rand_bits_2).view(np.complex128)) #list of awgn noise for each SNR ratio
    output_sign_2 = rand_bits_2_mod + awgn_exp[i].real #noise added to initial signal
    receiv_sign_2 = (output_sign_2 >= 0).astype(int) #denoise of received signal
    BER_exp.append(np.sum(receiv_sign_2 != rand_bits_2) / N_rand_bits_2) #BER calculation

#THEORETICAL
def q_bpam(a):
    return (1.0/math.sqrt(2*math.pi))*scipy.integrate.quad(lambda x: math.exp(-(x**2)/2), a, pow(10,2))[0]
BER_theor = []
for i in t_BER:
    BER_theor.append(q_bpam(np.sqrt(2*SNR_dB_lin(i))))

plt.semilogy(t_BER, BER_exp, color='r', marker='o', markersize=2, linestyle='', label='Experimental BER')
plt.semilogy(t_BER, BER_theor, marker='', linestyle='-', linewidth=1, label='Theoretical BER' )
plt.title('Experimental & theoretical BER curve')
plt.xlabel('$E_b/N_0(dB)$');plt.ylabel('BER ($P_e$)')
plt.grid(True)
plt.legend(loc='upper right')
##plt.savefig('ex3-photos/BER-diagram-B-PAM.png')
plt.show()


########### QUESTION 4 #########

#QPSK encoding
'''
00 -> s1
01 -> s2
11 -> s3
10 -> s4
'''

#### Α ΕΡΏΤΗΜΑ ####
#QPSK CONSTELLATION POINTS (00, 01, 11, 10)
qpsk_num_symbols = 100
qpsk_v_size = math.sqrt(E_b) #vector size of qpsk constellation
qpsk_const_points = np.random.randint(0, 4, qpsk_num_symbols) # 4 points generated
qpsk_const_degrees = qpsk_const_points*360/4.0 + 45 # 45, 135, 225, 315 degrees (π/4)
qpsk_const_radians = qpsk_const_degrees*np.pi/180.0 # sin() and cos() to calculate position for each point
qpsk_const_symbols = qpsk_v_size*np.cos(qpsk_const_radians) + qpsk_v_size*1j*np.sin(qpsk_const_radians) # QPSK vectors (complex)
plt.plot(np.real(qpsk_const_symbols), np.imag(qpsk_const_symbols), '.')
plt.grid(True)
plt.xlim(-qpsk_v_size-1, qpsk_v_size+1); plt.ylim(-qpsk_v_size-1, qpsk_v_size+1)
plt.title('Constellation of QPSK')   
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
##plt.savefig('ex4-photos/QPSK-constellation-46bits.png')
plt.show()

#### B ΕΡΏΤΗΜΑ ####
#AWGN CONSTELLATION WITH NOISE (5dB)
awgn_5 = np.random.normal(0, np.sqrt(No_5), 2*qpsk_num_symbols).view(np.complex128) #complex awgn (5dB)
qpsk_5_points = qpsk_const_symbols + awgn_5*math.sqrt(T_b)
plt.plot(np.real(qpsk_5_points), np.imag(qpsk_5_points), '.')
plt.grid(True) 
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
##plt.savefig('ex4-photos/QPSK-constellation-awgn-5dB.png')
plt.show()

#AWGN CONSTELLATION WITH NOISE (5dB)
awgn_15 = np.random.normal(0, np.sqrt(No_15), 2*qpsk_num_symbols).view(np.complex128) #complex awgn (15dB)
qpsk_15_points = qpsk_const_symbols + awgn_15*math.sqrt(T_b)
plt.plot(np.real(qpsk_15_points), np.imag(qpsk_15_points), '.')
plt.grid(True, which='both') 
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
##plt.savefig('ex4-photos/QPSK-constellation-awgn-15dB.png')
plt.show()


#### Γ ΕΡΏΤΗΜΑ ####

#BITSTREAM GENERATOR 
N_rand_bits_4 = 10**5 #number of random bits for each SNR ratio
rand_bits_4 = np.random.randint(2, size=(N_rand_bits_4)) #random bits generated (array)
rand_bits_4_str = ''.join([str(i) for i in rand_bits_4]) #random array of bits stringified

#BITSTREAM MODULATION - QPSK
rand_bits_qpsk_4 = []
for i in range(0, len(rand_bits_4_str), 2):
    x = rand_bits_4_str[i]+rand_bits_4_str[i+1]
    if (x=='00'):
        rand_bits_qpsk_4.append(0)
    elif (x=='01'):
        rand_bits_qpsk_4.append(1)
    elif (x=='11'):
        rand_bits_qpsk_4.append(2)
    elif (x=='10'):
        rand_bits_qpsk_4.append(3)

qpsk_exp_const_degrees = np.array(rand_bits_qpsk_4)*360/4.0 + 45  # 45, 135, 225, 315 degrees (π/4)
qpsk_exp_const_radians = qpsk_exp_const_degrees*np.pi/180.0 # sin() and cos() to calculate position for each point
qpsk_exp_const_symbols = qpsk_v_size*np.cos(qpsk_exp_const_radians) + qpsk_v_size*1j*np.sin(qpsk_exp_const_radians) # QPSK vectors (complex)

awgn_exp_4, output_sign_4, output_sign_4_radians, BER_exp_4 = [], [], [], []
for i in t_BER:
    #NOISE GENERATOR
    awgn_exp_4.append(np.random.normal(0, np.sqrt(No_exp[i]/2), 2*N_rand_bits_4//2).view(np.complex128)) #list of awgn noise for each SNR ratio (2* because of complex numbers , /2 because of qpsk modulation)
    output_sign_4.append(qpsk_exp_const_symbols + awgn_exp_4[i])
    #SIGNAL RECONSTRUCTION
    output_sign_4_radians.append(np.arctan2(np.imag(output_sign_4[i]), np.real(output_sign_4[i])))
    receiv_sign_4 = '' #reconstructed binary signal (string)
    for j in output_sign_4_radians[i]:
        if (j>=0 and j<np.pi/2):
            receiv_sign_4 += '00'
        elif (j>=np.pi/2 and j<np.pi):
            receiv_sign_4 += '01'
        elif (j>=-np.pi and j<-np.pi/2):
            receiv_sign_4 += '11'
        elif (j>=-np.pi/2 and j<0):
            receiv_sign_4 += '10'
    #BER CALCULATION
    error_bits = 0
    for j in range(0, len(receiv_sign_4)-1):
        if (receiv_sign_4[j] != rand_bits_4_str[j]):
            error_bits += 1
    BER_exp_4.append(error_bits / len(receiv_sign_4))

#THEORETICAL VALUES
BER_theor_4 = []
for i in t_BER:
    BER_theor_4.append(q_bpam(np.sqrt(SNR_dB_lin(i))))

#BER DIAGRAM
plt.semilogy(t_BER, BER_exp_4, color='r', marker='o', markersize=2, linestyle='', label='QPSK Experimental')
plt.semilogy(t_BER, BER_theor_4, marker='', linestyle='-', linewidth=1, label='QPSK Theoretical')
plt.semilogy(t_BER, BER_exp, color='g', marker='o', markersize=2, linestyle='', label='BPSK Experimental')
plt.title('Experimental & theoretical BER curve')
plt.xlabel('$E_b/N_0(dB)$');plt.ylabel('BER ($P_e$)')
plt.legend(loc='upper right')
plt.grid(True)
##plt.savefig('ex4-photos/BER-diagram-QPSK.png')
plt.show()


#### Δ ΕΡΏΤΗΜΑ ####

##PARAMETERS
if (A_bit%2):
    file_name = 'shannon_odd.txt'
else:
    file_name = 'shannon_even.txt'

#I) 
#STRING TO BINARY
def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int(binascii.hexlify(text.encode(encoding, errors)), 16))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

#BINARY TO STRING
def bits_to_text(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'

file = open('samples/'+file_name, 'r')
#text file stringified
file_string = ''
for i in file:
    file_string += i

file_ascii = []
for i in file_string:
    file_ascii.append(ord(i))

#TEXT CONVERTED TO BINARY
file_bin = text_to_bits(file_string)

#II)
t_ascii = np.arange(0, len(file_ascii))
plt.plot(t_ascii, file_ascii, linestyle='-', linewidth=0.5, label='Digital Signal of Text File')
plt.title(file_name)
plt.xlabel('Characters'); plt.ylabel('ASCII')
plt.legend(loc='upper left')
plt.ylim(30, 130)
##plt.savefig('ex4-photos/Text-quantization.png')
plt.show()

#III)
'''
00 -> s1(t) = V*cos(2*pi*fc*t)
01 -> s2(t) = V*sin(2*pi*fc*t)
11 -> s3(t) = -V*cos(2*pi*fc*t)
10 -> s4(t) = -V*sin(2*pi*fc*t)
'''

#BITSTREAM OF INITIAL FILE -> file_bin
#QPSK MODULATION -> file_bin_qpsk
file_bin_qpsk = []
for i in range(0, len(file_bin), 2):
    x = file_bin[i]+file_bin[i+1]
    if (x=='00'):
        file_bin_qpsk.append(0)
    elif (x=='01'):
        file_bin_qpsk.append(1)
    elif (x=='11'):
        file_bin_qpsk.append(2)
    elif (x=='10'):
        file_bin_qpsk.append(3)

#4 points represented as vectors
qpsk_text_const_degrees = np.array(file_bin_qpsk)*360/4.0 # 0, 90, 180, 270 degrees (0)
qpsk_text_const_radians = qpsk_text_const_degrees*np.pi/180.0 # sin() and cos() to calculate position for each point
qpsk_text_const_symbols = np.cos(qpsk_text_const_radians) + 1j*np.sin(qpsk_text_const_radians) # QPSK vectors (complex)

plt.plot(np.real(qpsk_text_const_symbols), np.imag(qpsk_text_const_symbols), '.')
plt.grid(True)
plt.xlim(-qpsk_v_size, qpsk_v_size); plt.ylim(-qpsk_v_size, qpsk_v_size)
plt.title('Constellation of QPSK of text file')   
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
##plt.savefig('ex4-photos/Text-QPSK-constellation.png')
plt.show()

#IV)
E_b_text = pow(1, 2)*T_b 
No_5_text = E_b_text / SNR_dB_lin(5)
No_15_text = E_b_text / SNR_dB_lin(15) 

#AWGN NOISE FOR SNR=5dB
text_awgn_5 = np.random.normal(0, np.sqrt(No_5_text), 2*len(file_bin_qpsk)).view(np.complex128) #complex awgn (5dB)
qpsk_5_points = qpsk_text_const_symbols + text_awgn_5 #signal + noize(5dB)

#AWGN NOISE FOR SNR=15dB
text_awgn_15 = np.random.normal(0, np.sqrt(No_15_text), 2*len(file_bin_qpsk)).view(np.complex128) #complex awgn (15dB)
qpsk_15_points = qpsk_text_const_symbols + text_awgn_15 #signal + noize(15dB)

#V)
#CONSTELLATION DIAGRAM WITH AWGN NOISE (5dB)
plt.plot(np.real(qpsk_5_points), np.imag(qpsk_5_points), '.')
plt.grid(True) 
plt.title('QPSK Constellation for Es/No=5')
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
##plt.savefig('ex4-photos/Text-QPSK-constellation-awgn-5dB.png')
plt.show()

#CONSTELLATION DIAGRAM WITH AWGN NOISE (15dB)
plt.plot(np.real(qpsk_15_points), np.imag(qpsk_15_points), '.')
plt.grid(True, which='both') 
plt.title('QPSK Constellation for Es/No=15')
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
##plt.savefig('ex4-photos/Text-QPSK-constellation-awgn-15dB.png')
plt.show()

#SIGNAL RECONSTRUCTION FOR SNR=5dB
qpsk_recon_radians_5 = np.arctan2(np.imag(qpsk_5_points), np.real(qpsk_5_points))
file_bin_recon_5 = ''
for i in qpsk_recon_radians_5:
    if (i>=-np.pi/4 and i<np.pi/4):
        file_bin_recon_5 += '00'
    elif (i>=np.pi/4 and i<np.pi*3/4):
        file_bin_recon_5 += '01'
    elif (i>=np.pi*3/4 or i<-np.pi*3/4):
        file_bin_recon_5 += '11'
    elif (i>=-np.pi*3/4 and i<-np.pi/4):
        file_bin_recon_5 += '10'

#SIGNAL RECONSTRUCTION FOR SNR=15dB
qpsk_recon_radians_15 = np.arctan2(np.imag(qpsk_15_points), np.real(qpsk_15_points))
file_bin_recon_15 = ''
for i in qpsk_recon_radians_15:
    if (i>=-np.pi/4 and i<np.pi/4):
        file_bin_recon_15 += '00'
    elif (i>=np.pi/4 and i<np.pi*3/4):
        file_bin_recon_15 += '01'
    elif (i>=np.pi*3/4 or i<-np.pi*3/4):
        file_bin_recon_15 += '11'
    elif (i>=-np.pi*3/4 and i<-np.pi/4):
        file_bin_recon_15 += '10'

#VI)
#BER calculation for SNR=5 & SNR=15

#EXPERIMENTAL 
count_err_5 = 0.0
count_err_15 = 0.0
for i in range(0, len(file_bin)-1):
    if (file_bin[i] != file_bin_recon_5[i]): 
        count_err_5 += 1
    if (file_bin[i] != file_bin_recon_15[i]):
        count_err_15 += 1

BER_exp_5 = count_err_5 / len(file_bin)
BER_exp_15 = count_err_15 / len(file_bin)

#THEORETICAL
BER_theor_5 = q_bpam(np.sqrt(SNR_dB_lin(5)))
BER_theor_15 = q_bpam(np.sqrt(SNR_dB_lin(15)))

print('Experimental BER for text file (SNR=5dB) : '+ str(BER_exp_5))
print('Theoretical BER for text file (SNR=5dB) : '+ str(BER_theor_5))
print('Experimental BER for text file (SNR=15dB) : '+ str(BER_exp_15))
print('Theoretical BER for text file (SNR=15dB) : '+ str(BER_theor_15))

#VII)

#PARAMETERS
if (A_bit % 2):
    text_export_name_5dB = 'shannon_odd_export_5dB'
    text_export_name_15dB = 'shannon_odd_export_15dB'
else:
    text_export_name_5dB = 'shannon_even_export_5dB'
    text_export_name_15dB = 'shannon_even_export_15dB'

#File reconstruction for SNR=5dB
file_bin_recon_5_arr = [file_bin_recon_5[i:i+8] for i in range(0, len(file_bin_recon_5), 8)]
file_string_recon_5 = ''
for i in file_bin_recon_5_arr:
    file_string_recon_5 += chr(int(i, 2))
new_file = open(text_export_name_5dB+'.txt', 'w', encoding='utf-8')
new_file.write(file_string_recon_5)
new_file.close()

#File reconstruction for SNR=5dB
file_bin_recon_15_arr = [file_bin_recon_15[i:i+8] for i in range(0, len(file_bin_recon_15), 8)]
file_string_recon_15 = ''
for i in file_bin_recon_15_arr:
    file_string_recon_15 += chr(int(i, 2))
new_file = open(text_export_name_15dB+'.txt', 'w', encoding='utf-8')
new_file.write(file_string_recon_15)
new_file.close()


########### QUESTION 5 #########

#### Α ΕΡΏΤΗΜΑ ####

##PARAMETERS
if (A_bit%2):
    audio_file_name = 'soundfile1_lab2.wav'
else:
    audio_file_name = 'soundfile2_lab2.wav'

audio_samplerate, audio_file = wavfile.read('samples/'+audio_file_name)
audio_file_length = audio_file.shape[0] / audio_samplerate
t_audio_file = np.linspace(0., audio_file_length, audio_file.shape[0])

plt.plot(t_audio_file, audio_file, label='signed 16-bit PCM Mono 44100 Hz')
plt.title('Audio Signal '+ audio_file_name)
plt.xlabel("Time [s]"); plt.ylabel("Amplitude")
plt.legend()
#plt.savefig('ex5-photos/Audio.png')
plt.show()


#### B ΕΡΏΤΗΜΑ ####

##PARAMETERS
audio_quant_bits = 8
audio_q_levels = 2**audio_quant_bits              #quantization levels (int)
audio_q_levels_top = audio_q_levels/2             #quantization levels on one side (int)             
audio_s_max = max(abs(audio_file))                #get the max value
audio_delta = (2*audio_s_max)/(audio_q_levels-1)

#QUANTIZATION
audio_file_quant = np.copy(audio_file) #np.copy() copies audio_file array without reference
audio_file_new = np.copy(audio_file)
for i in range(0,audio_file.size):
    audio_file_quant[i] = int(math.floor(round(audio_file[i],4)/audio_delta)) #quantized levels (int)
    audio_file_new[i] = audio_delta*(audio_file_quant[i])+audio_delta/2 #mid-riser quantization

#PLOT FOR QUANTIZED SIGNAL
gray_code_ex5 = gray_code(audio_quant_bits)
plt.plot(t_audio_file, audio_file_quant, label='8-bit quantized')
plt.title('Audio quantized with mid riser (8 bits)')   
plt.xlabel('Time (sec)'); plt.ylabel("Quantized value")
plt.legend(loc='upper left')
#plt.savefig('ex5-photos/Audio-quantized(8bits).png')
plt.show()


#### Γ ΕΡΏΤΗΜΑ ####

#AUDIO FILE CONVERTED TO BINARY STRING
audio_bin = ''
for i in range(0,audio_file.size):
    audio_bin += gray_code_ex5[int(audio_file_quant[i] + audio_q_levels/2)]

audio_bin_qpsk = []
for i in range(0, len(audio_bin), 2):
    x = audio_bin[i]+audio_bin[i+1]
    if (x=='00'):
        audio_bin_qpsk.append(0)
    elif (x=='01'):
        audio_bin_qpsk.append(1)
    elif (x=='11'):
        audio_bin_qpsk.append(2)
    elif (x=='10'):
        audio_bin_qpsk.append(3)

qpsk_audio_const_degrees = np.array(audio_bin_qpsk)*360/4.0 # 0, 90, 180, 270 degrees (π/4)
qpsk_audio_const_radians = qpsk_audio_const_degrees*np.pi/180.0 # sin() and cos() to calculate position for each point
qpsk_audio_const_symbols = np.cos(qpsk_audio_const_radians) + 1j*np.sin(qpsk_audio_const_radians) # QPSK vectors (complex)

plt.plot(np.real(qpsk_audio_const_symbols), np.imag(qpsk_audio_const_symbols), '.')
plt.grid(True)
plt.xlim(-qpsk_v_size, qpsk_v_size); plt.ylim(-qpsk_v_size, qpsk_v_size)
plt.title('Constellation of QPSK of audio file')   
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
#plt.savefig('ex5-photos/Audio-QPSK.png')
plt.show()


#### Δ ΕΡΏΤΗΜΑ ####
E_b_audio = pow(1, 2)*T_b 
No_4_audio = E_b_audio / SNR_dB_lin(4)
No_14_audio = E_b_audio / SNR_dB_lin(14) 

#AWGN NOISE FOR SNR=4dB
audio_awgn_4 = np.random.normal(0, np.sqrt(No_4_audio), 2*len(audio_bin_qpsk)).view(np.complex128) #complex awgn (5dB)
qpsk_4_points = qpsk_audio_const_symbols + audio_awgn_4 #signal + noize(4dB)

#AWGN NOISE FOR SNR=14dB
audio_awgn_14 = np.random.normal(0, np.sqrt(No_14_audio), 2*len(audio_bin_qpsk)).view(np.complex128) #complex awgn (15dB)
qpsk_14_points = qpsk_audio_const_symbols + audio_awgn_14 #signal + noize(14dB)


#### Ε ΕΡΏΤΗΜΑ ####

#QPSK CONSTELLATION DIAGRAM WITH AWGN NOISE (4dB)
plt.plot(np.real(qpsk_4_points), np.imag(qpsk_4_points), '.')
plt.grid(True) 
plt.title('QPSK Constellation for Es/No=4')
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
#plt.savefig('ex5-photos/Audio-QPSK-4dB.png')
plt.show()

#QPSK CONSTELLATION DIAGRAM WITH AWGN NOISE (14dB)
plt.plot(np.real(qpsk_14_points), np.imag(qpsk_14_points), '.')
plt.grid(True, which='both') 
plt.title('QPSK Constellation for Es/No=14')
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
#plt.savefig('ex5-photos/Audio-QPSK-14dB.png')
plt.show()

#SIGNAL RECONSTRUCTION FOR SNR=4dB
qpsk_recon_radians_4 = np.arctan2(np.imag(qpsk_4_points), np.real(qpsk_4_points))
audio_bin_recon_4 = ''
for i in qpsk_recon_radians_4:
    if (i>=-np.pi/4 and i<np.pi/4):
        audio_bin_recon_4 += '00'
    elif (i>=np.pi/4 and i<np.pi*3/4):
        audio_bin_recon_4 += '01'
    elif (i>=np.pi*3/4 or i<-np.pi*3/4):
        audio_bin_recon_4 += '11'
    elif (i>=-np.pi*3/4 and i<-np.pi/4):
        audio_bin_recon_4 += '10'

#SIGNAL RECONSTRUCTION FOR SNR=14dB
qpsk_recon_radians_14 = np.arctan2(np.imag(qpsk_14_points), np.real(qpsk_14_points))
audio_bin_recon_14 = ''
for i in qpsk_recon_radians_14:
    if (i>=-np.pi/4 and i<np.pi/4):
        audio_bin_recon_14 += '00'
    elif (i>=np.pi/4 and i<np.pi*3/4):
        audio_bin_recon_14 += '01'
    elif (i>=np.pi*3/4 or i<-np.pi*3/4):
        audio_bin_recon_14 += '11'
    elif (i>=-np.pi*3/4 and i<-np.pi/4):
        audio_bin_recon_14 += '10'

#### ΣΤ ΕΡΏΤΗΜΑ ####
#BER calculation for SNR=5 & SNR=15

#EXPERIMENTAL
count_err_4 = 0.0
count_err_14 = 0.0
for i in range(0, len(audio_bin)-1):
    if (audio_bin[i] != audio_bin_recon_4[i]): 
        count_err_4 += 1
    if (audio_bin[i] != audio_bin_recon_14[i]):
        count_err_14 += 1

BER_exp_4 = count_err_4 / len(audio_bin)
BER_exp_14 = count_err_14 / len(audio_bin)

#THEORETICAL
BER_theor_4 = q_bpam(np.sqrt(SNR_dB_lin(4)))
BER_theor_14 = q_bpam(np.sqrt(SNR_dB_lin(14)))

print('Experimental BER for audio file (SNR=4dB) : '+ str(BER_exp_4))
print('Theoretical BER for audio file (SNR=4dB) : '+ str(BER_theor_4))
print('Experimental BER for audio file (SNR=14dB) : '+ str(BER_exp_14))
print('Theoretical BER for audio file (SNR=14dB) : '+ str(BER_theor_14))

#### Ζ ΕΡΏΤΗΜΑ ####

#PARAMETERS
if (A_bit % 2):
    audio_export_name_4dB = 'soundfile1_export_4dB'
    audio_export_name_14dB = 'soundfile1_export_14dB'
else:
    audio_export_name_4dB = 'soundfile2_export_4dB'
    audio_export_name_14dB = 'soundfile2_export_14dB'

#SIGNAL AMPLITUDE RECONSTRUCTION FROM BITS
#FOR SNR = 4dB
audio_bin_recon_4_arr = [audio_bin_recon_4[i:i+8] for i in range(0, len(audio_bin_recon_4), 8)] #bit-stream to 8-bit list elements
audio_file_quant_recon_4 = [] #quantized levels after reconstruction (int)
audio_file_new_recon_4 = [] #reconstructed signal
for i in range (0, len(audio_bin_recon_4_arr)):
    audio_file_quant_recon_4.append(gray_code_ex5.index(audio_bin_recon_4_arr[i]) - audio_q_levels/2)
    audio_file_new_recon_4.append(audio_delta*(audio_file_quant_recon_4[i]) + audio_delta/2)

wavfile.write(audio_export_name_4dB+'.wav', audio_samplerate, np.array(audio_file_new_recon_4, dtype=np.uint8))
wavfile.write(audio_export_name_4dB+'-16bit.wav', audio_samplerate, np.array(audio_file_new_recon_4, dtype=np.int16))

#FOR SNR = 14dB
audio_bin_recon_14_arr = [audio_bin_recon_14[i:i+8] for i in range(0, len(audio_bin_recon_14), 8)] #bit-stream to 8-bit list elements
audio_file_quant_recon_14 = [] #quantized levels after reconstruction (int)
audio_file_new_recon_14 = [] #reconstructed signal
for i in range (0, len(audio_bin_recon_14_arr)):
    audio_file_quant_recon_14.append(gray_code_ex5.index(audio_bin_recon_14_arr[i]) - audio_q_levels/2)
    audio_file_new_recon_14.append(audio_delta*(audio_file_quant_recon_14[i]) + audio_delta/2)

wavfile.write(audio_export_name_14dB+'.wav', audio_samplerate, np.array(audio_file_new_recon_14, dtype=np.uint8))
wavfile.write(audio_export_name_14dB+'-16bit.wav', audio_samplerate, np.array(audio_file_new_recon_14, dtype=np.int16))
