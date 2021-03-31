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

##PARAMETERS
Fm = 3000 #kHz
Tm = 1 / Fm #sec
A = 1 #V
AM = 3 
N_periods = 4 #periods displayed

Samples_per_period = 2000 #number of samples per period
N_samples = N_periods * Samples_per_period + 1 #total number of samples (in linspace)
Timestep = 1.0 / (float(Fm * Samples_per_period)) #sample spacing

A_bit = Fm/1000 #(V) Amplitude of bit stream
T_b = 0.5 #(sec) bit duration 
N_rand_bits = 46 #number of random bits generated
E_b = pow(A_bit, 2)*T_b 

rand_bits = np.random.randint(2, size=(N_rand_bits)) #generate random bits [0,1]

def SNR_dB_lin(snr_ratio):
    return 10**(snr_ratio / 10)

No_5 = E_b / SNR_dB_lin(5) #Conversion from dB to linear scale
No_15 = E_b / SNR_dB_lin(15) #Conversion from dB to linear scale

qpsk_v_size = math.sqrt(E_b)

#QPSK encoding
'''
00 -> s1
01 -> s2
11 -> s3
10 -> s4
'''

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


#### Α ΕΡΏΤΗΜΑ ####

##PARAMETERS
if (A_bit%2):
    audio_file_name = 'soundfile1_lab2.wav'
else:
    audio_file_name = 'soundfile2_lab2.wav'

audio_samplerate, audio_file = wavfile.read('../samples/'+audio_file_name)
audio_file_length = audio_file.shape[0] / audio_samplerate
t_audio_file = np.linspace(0., audio_file_length, audio_file.shape[0])

plt.plot(t_audio_file, audio_file, label='signed 16-bit PCM Mono 44100 Hz')
plt.title('Audio Signal '+ audio_file_name)
plt.xlabel("Time [s]"); plt.ylabel("Amplitude")
plt.legend()
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
# plt.vlines(t_20, [0], quant_signal, linewidth=0.8, colors="b")
# plt.yticks(np.arange(-q_levels/2, q_levels/2, 1), gray_code_ex2)
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) #for automatic x-scale (10^-3)
plt.plot(t_audio_file, audio_file_quant, label='8-bit quantized')
plt.title('Audio quantized with mid riser (8 bits)')   
plt.xlabel('Time (sec)')
# plt.ylabel('Gray Code')
plt.legend(loc='upper left')
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
# qpsk_text_const_radians = np.radians(qpsk_text_const_degrees)
qpsk_audio_const_symbols = np.cos(qpsk_audio_const_radians) + 1j*np.sin(qpsk_audio_const_radians) # QPSK vectors (complex)

plt.plot(np.real(qpsk_audio_const_symbols), np.imag(qpsk_audio_const_symbols), '.')
plt.grid(True)
plt.xlim(-qpsk_v_size, qpsk_v_size); plt.ylim(-qpsk_v_size, qpsk_v_size)
plt.title('Constellation of QPSK of audio file')   
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
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
plt.show()

#QPSK CONSTELLATION DIAGRAM WITH AWGN NOISE (14dB)
plt.plot(np.real(qpsk_14_points), np.imag(qpsk_14_points), '.')
plt.grid(True, which='both') 
plt.title('QPSK Constellation for Es/No=14')
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
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
count_err_4 = 0.0
count_err_14 = 0.0
for i in range(0, len(audio_bin)-1):
    if (audio_bin[i] != audio_bin_recon_4[i]): 
        count_err_4 += 1
    if (audio_bin[i] != audio_bin_recon_14[i]):
        count_err_14 += 1

BER_exp_4 = count_err_4 / len(audio_bin)
BER_exp_14 = count_err_14 / len(audio_bin)

print('Experimental BER for audio file (SNR=4dB) : '+ str(BER_exp_4))
print('Experimental BER for audio file (SNR=14dB) : '+ str(BER_exp_14))

#### Ζ ΕΡΏΤΗΜΑ ####

#SIGNAL AMPLITUDE RECONSTRUCTION FROM BITS

#FOR SNR = 4dB
audio_bin_recon_4_arr = [audio_bin_recon_4[i:i+8] for i in range(0, len(audio_bin_recon_4), 8)] #bit-stream to 8-bit list elements
audio_file_quant_recon_4 = [] #quantized levels after reconstruction (int)
audio_file_new_recon_4 = [] #reconstructed signal
for i in range (0, len(audio_bin_recon_4_arr)):
    audio_file_quant_recon_4.append(gray_code_ex5.index(audio_bin_recon_4_arr[i]) - audio_q_levels/2)
    audio_file_new_recon_4.append(audio_delta*(audio_file_quant_recon_4[i]) + audio_delta/2)

wavfile.write("audio_4dB.wav", audio_samplerate, np.array(audio_file_new_recon_4, dtype=np.uint8))

#FOR SNR = 14dB
audio_bin_recon_14_arr = [audio_bin_recon_14[i:i+8] for i in range(0, len(audio_bin_recon_14), 8)] #bit-stream to 8-bit list elements
audio_file_quant_recon_14 = [] #quantized levels after reconstruction (int)
audio_file_new_recon_14 = [] #reconstructed signal
for i in range (0, len(audio_bin_recon_14_arr)):
    audio_file_quant_recon_14.append(gray_code_ex5.index(audio_bin_recon_14_arr[i]) - audio_q_levels/2)
    audio_file_new_recon_14.append(audio_delta*(audio_file_quant_recon_14[i]) + audio_delta/2)

wavfile.write("audio_14dB.wav", audio_samplerate, np.array(audio_file_new_recon_14, dtype=np.uint8))
