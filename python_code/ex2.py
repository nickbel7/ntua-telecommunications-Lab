import numpy as np                      #for arrays and matrices
import matplotlib.pyplot as plt         #for ploting
import scipy                            #for math calculations
from mpl_toolkits.mplot3d import Axes3D #for 3D ploting
import math                             #for math constants
from matplotlib import collections as matcoll
from scipy import signal                #for signal analysis
from scipy import fftpack               #for fourier spectrum
from scipy.fftpack import fft

########### QUESTION 2 #########

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
#SIGNAL (20fm)
t_20 = np.linspace(0, N_periods*Tm, 4*20+1)
y_20 = A*np.cos(2*math.pi*Fm*t_20)*np.cos(2*math.pi*(AM+2)*Fm*t_20)

# plt.vlines(t_20, [0], y_20, linewidth=0.8, colors="b")
# plt.plot(t_20, y_20, '.')
# plt.show()

#######################################
bits = 5                      #bits for quantization
q_levels = 2**bits              #quantization levels
q_levels_top = q_levels/2       #quantization levels on one side               
s_max = max(abs(y_20))          #get the max value
delta = (2*s_max)/(q_levels-1)  #step size

#QUANTIZATION
quant_signal = np.copy(y_20) #np.copy() copies y_20 array without reference
y_20_new = np.copy(y_20)
for i in range(0,y_20.size):
    quant_signal[i] = int(math.floor(round(y_20[i],4)/delta)) #quantized levels (int)
    y_20_new[i] = delta*(quant_signal[i])+delta/2 #mid-riser quantization
    # print(str(y_20[i]) +' : '+ str(y_20_new[i]))
    # print(quant_signal[i])

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
# plt.show()
plt.figure()


#### Β ΕΡΏΤΗΜΑ ####
#Ι) -> variance (τυπική απόκλιση) for 10 samples 
error_10 = y_20[0:10]-y_20_new[0:10]
var_10 = (1/10)*sum(map(lambda x:x*x,error_10))
print('Variance for first \n10 samples : '+ str(var_10))

#ΙI) -> variance (τυπική απόκλιση) for 20 samples
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
Bitstream = '' #bit stream of output (string)
polar_nrz = []
for i in range(0, 20):
    Bitstream += gray_code_ex2[int(quant_signal[i]+q_levels/2)] #creates string of bitstream
#Option A
samples_per_bit = 100
for i in range(0, len(Bitstream)):
    for j in range(0, samples_per_bit):
        polar_nrz.append(int(Bitstream[i])) #appends bits to array
#Option B
# polar_nrz = list(Bitstream)


t_bit_20 = np.linspace(0, 0.001*bits*20, samples_per_bit*bits*20, endpoint=False)
plt.plot(t_bit_20, Fm/1000*signal.square(2*math.pi*t_bit_20, duty=polar_nrz[0:samples_per_bit*bits*20]), label='POLAR NRZ')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.title('Bit stream of quantized signal (fs1=20fm)')   
plt.xlabel('Time (sec)'); plt.ylabel('Amplitude (V)')
plt.legend(loc='upper left')
plt.show()