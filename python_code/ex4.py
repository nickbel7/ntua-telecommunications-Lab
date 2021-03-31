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
E_b = pow(A_bit, 2)*T_b 

rand_bits = np.random.randint(2, size=(N_rand_bits)) #generate random bits [0,1]

def SNR_dB_lin(snr_ratio):
    return 10**(snr_ratio / 10)

No_5 = E_b / SNR_dB_lin(5) #Conversion from dB to linear scale
No_15 = E_b / SNR_dB_lin(15) #Conversion from dB to linear scale

#QPSK encoding
'''
00 -> s1
01 -> s2
11 -> s3
10 -> s4
'''
qpsk_num_symbols = 100
#QPSK CONSTELLATION POINTS (00, 01, 11, 10)
qpsk_v_size = math.sqrt(E_b) #vector size of qpsk constellation
qpsk_const_points = np.random.randint(0, 4, qpsk_num_symbols) # 0 to 3
# qpsk_const_points = np.arange(0,4)
qpsk_const_degrees = qpsk_const_points*360/4.0 + 45 # 45, 135, 225, 315 degrees
qpsk_const_radians = qpsk_const_degrees*np.pi/180.0 # sin() and cos() takes in radians
qpsk_const_symbols = qpsk_v_size*np.cos(qpsk_const_radians) + qpsk_v_size*1j*np.sin(qpsk_const_radians) # this produces our QPSK complex symbols
plt.plot(np.real(qpsk_const_symbols), np.imag(qpsk_const_symbols), '.')
plt.grid(True)
plt.xlim(-qpsk_v_size-1, qpsk_v_size+1); plt.ylim(-qpsk_v_size-1, qpsk_v_size+1)
plt.title('Constellation of QPSK')   
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
plt.show()
# plt.figure()


# x_bpam = [-math.sqrt(E_b)/math.sqrt(2), math.sqrt(E_b)/math.sqrt(2)]
# y_bpam = [0, 0]
# plt.scatter(x_bpam,y_bpam)
# plt.ylim([-0.5, 0.5])
# plt.grid(True, which='both')
# plt.title('Constellation of B-PAM')   
# plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
# plt.show()
# plt.figure()

#### B ΕΡΏΤΗΜΑ ####
#AWGN CONSTELLATION WITH NOISE (5dB)
awgn_5 = np.random.normal(0, np.sqrt(No_5), 2*qpsk_num_symbols).view(np.complex128) #complex awgn (5dB)
qpsk_5_points = qpsk_const_symbols + awgn_5*math.sqrt(T_b)
plt.plot(np.real(qpsk_5_points), np.imag(qpsk_5_points), '.')
plt.grid(True) 
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
plt.show()

#AWGN CONSTELLATION WITH NOISE (15dB)
awgn_15 = np.random.normal(0, np.sqrt(No_15), 2*qpsk_num_symbols).view(np.complex128) #complex awgn (15dB)
qpsk_15_points = qpsk_const_symbols + awgn_15*math.sqrt(T_b)
plt.plot(np.real(qpsk_15_points), np.imag(qpsk_15_points), '.')
plt.grid(True, which='both') 
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
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
plt.show()
# rand_bits_2_mod = rand_bits_2*2*A_bit-A_bit #rand bits modulated (1->A_bit, 0->-A_bit)
# t_BER = np.arange(0, 16) #linear space

#EXPERIMENTAL
# No_exp_4, awgn_exp_4, BER_exp_4 = [], [], []
# for i in t_BER:
#     No_exp.append(E_b / SNR_dB_lin(i))
#     awgn_exp.append(np.random.normal(0, np.sqrt(No_exp[i]), 2*N_rand_bits_2).view(np.complex128))
#     output_sign = rand_bits_2_mod + awgn_exp[i].real
#     receiv_sign = (output_sign >= 0).astype(int)
#     BER_exp.append(np.sum(receiv_sign != rand_bits_2) / N_rand_bits_2)

# #THEORETICAL
# # BER_theor = scipy.special.erfc(np.sqrt(SNR_dB_lin(t_BER)))
# def q_bpam(a):
#     return (1.0/math.sqrt(2*math.pi))*scipy.integrate.quad(lambda x: math.exp(-(x**2)/2), a, pow(10,2))[0]
# BER_theor = []
# for i in t_BER:
#     BER_theor.append(q_bpam(np.sqrt(2*SNR_dB_lin(i))))

# plt.semilogy(t_BER, BER_exp, color='r', marker='o', markersize=2, linestyle='')
# plt.semilogy(t_BER, BER_theor, marker='', linestyle='-', linewidth=1 )
# plt.title('Experimental & theoretical BER curve')
# plt.xlabel('$E_b/N_0(dB)$');plt.ylabel('BER ($P_e$)')
# plt.grid(True)
# plt.show()


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

file = open('../samples/'+file_name, 'r')
#text file stringified
file_string = ''
for i in file:
    file_string += i

file_ascii = []
for i in file_string:
    file_ascii.append(ord(i)) #ord returns the unicode of a character (ex. h -> 104)

#print binary 
file_bin = text_to_bits(file_string)

#print text reconstructed
file_string_new = bits_to_text(file_bin)
file.close()

# new_file = open('samples/new.txt', 'w')
# new_file.write(file_string_new)
# new_file.close()

#II)
t_ascii = np.arange(0, len(file_ascii))
plt.plot(t_ascii, file_ascii, linestyle='-', linewidth=0.5, label='Digital Signal of Text File')
plt.title(file_name)
plt.xlabel('Characters'); plt.ylabel('ASCII')
plt.legend(loc='upper left')
plt.ylim(30, 130)
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

#Binary digital signal converted to 4 values
# qpsk_text_const_points = np.random.randint(0, 4, int(len(file_bin)/2)) # many random points generated in 4 positions

#4 points represented as vectors
qpsk_text_const_degrees = np.array(file_bin_qpsk)*360/4.0 # 0, 90, 180, 270 degrees (π/4)
qpsk_text_const_radians = qpsk_text_const_degrees*np.pi/180.0 # sin() and cos() to calculate position for each point
# qpsk_text_const_radians = np.radians(qpsk_text_const_degrees)
qpsk_text_const_symbols = np.cos(qpsk_text_const_radians) + 1j*np.sin(qpsk_text_const_radians) # QPSK vectors (complex)

plt.plot(np.real(qpsk_text_const_symbols), np.imag(qpsk_text_const_symbols), '.')
plt.grid(True)
plt.xlim(-qpsk_v_size, qpsk_v_size); plt.ylim(-qpsk_v_size, qpsk_v_size)
plt.title('Constellation of QPSK of text file')   
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
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
plt.show()

#CONSTELLATION DIAGRAM WITH AWGN NOISE (15dB)
plt.plot(np.real(qpsk_15_points), np.imag(qpsk_15_points), '.')
plt.grid(True, which='both') 
plt.title('QPSK Constellation for Es/No=15')
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
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
#BER for 5,15dB
# file_bin initial binary string
# file_bin_recon_5 binary string after SNR=5
# file_bin_recon_15 binary string after SNR=15
#EXPERIMENTAL 
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
#File reconstruction for SNR=5dB
file_bin_recon_5_arr = [file_bin_recon_5[i:i+8] for i in range(0, len(file_bin_recon_5), 8)]
file_string_recon_5 = ''
for i in file_bin_recon_5_arr:
    file_string_recon_5 += chr(int(i, 2))
    # print(chr(int(i, 2)))

# file_string_recon_5 = bits_to_text(file_bin_recon_5)
new_file = open('shannon_5dB.txt', 'w', encoding='utf-8')
new_file.write(file_string_recon_5)
new_file.close()

#File reconstruction for SNR=15dB
file_string_recon_15 = bits_to_text(file_bin_recon_15)
new_file = open('shannon_15dB.txt', 'w', encoding='utf-8')
new_file.write(file_string_recon_15)
new_file.close()
