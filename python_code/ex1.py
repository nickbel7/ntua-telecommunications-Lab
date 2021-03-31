import numpy as np                      #for arrays and matrices
import matplotlib.pyplot as plt         #for ploting
import scipy                            #for math calculations
from mpl_toolkits.mplot3d import Axes3D #for 3D ploting
import math                             #for math constants
from matplotlib import collections as matcoll
from scipy import signal                #for signal analysis
from scipy.fftpack import fft           #for fourier spectrum

########### QUESTION 1 #########

##PARAMETERS
Fm = 3000 #kHz
Tm = 1 / 3000 #sec
A = 1 #V
AM = 3 
N_periods = 4 #periods displayed

#INITIAL SIGNAL
t = np.linspace(0, N_periods*Tm, 8000+1)
y = A*np.cos(2*math.pi*Fm*t)*np.cos(2*math.pi*(AM+2)*Fm*t)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) #for automatic x-scale (10^-3)
plt.plot(t, y, '-', marker="", markersize=4) #for markers
plt.show()

#Α ΕΡΏΤΗΜΑ
#version 1
def sign_sampling(coeff):
    t_freq = np.linspace(0, N_periods*Tm, N_periods*coeff+1)
    y_freq = A*np.cos(2*math.pi*Fm*t_freq)*np.cos(2*math.pi*(AM+2)*Fm*t_freq)
    plt.vlines(t_freq, [0], y_freq, linewidth=0.3)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) #for automatic x-scale (10^-3)
    plt.plot(t_freq, y_freq, '.', marker=".", markersize=4)
    # plt.plot(t, y, '-', t_freq, y_freq, '.', N_periods*Tm, y[0])
    plt.show()

#version 2
def sign_sampling_2(coeff):
    t_freq = np.linspace(0, N_periods*Tm, N_periods*coeff)
    y_freq = A*np.cos(2*math.pi*Fm*t_freq)*np.cos(2*math.pi*(AM+2)*Fm*t_freq)
    plt.plot(t, y, '-', t_freq, y_freq, '.', N_periods*Tm, y[0])
    plt.show()

#Ι) -> fs1=20fm
##### version 1 #####
# t_20 = np.linspace(0, 4/3000, 80, endpoint=False)
# y_20 = signal.resample(y, 80)
# plt.vlines(t_20, [0], y_20, linewidth=0.3)
# plt.plot(t_20, y_20, '.', marker=".", markersize=4)
# plt.show()

##### version 2 #####
# t_20 = np.linspace(0,4/3000,80)
# y_20 = A*np.cos(2*math.pi*3000*t_20)*np.cos(2*math.pi*(AM+2)*3000*t_20)
# plt.vlines(t_20, [0], y_20, linewidth=0.3)
# plt.plot(t_20, y_20, '.', marker=".", markersize=4)
# plt.show()

##### version 3 #####
sign_sampling(20)

##### version 4 #####
# t = np.arange(0, 4/3000, 1/(100*3000))
# y = A*np.cos(2*math.pi*3000*t)*np.cos(2*math.pi*(AM+2)*3000*t)
# plt.stem(t, y, use_line_collection=True)
# plt.show()


#ΙΙ) -> fs2=100fm 
##### version 1 #####
# t_100 = np.linspace(0, 4/3000, 400, endpoint=False)
# y_100 = signal.resample(y, 400)
# plt.vlines(t_100, [0], y_100, linewidth=0.3)
# plt.plot(t_100, y_100, '.', marker=".", markersize=4)
# plt.show()

##### version 2 #####
# t_100 = np.linspace(0,4/3000,400)
# y_100 = A*np.cos(2*math.pi*3000*t_100)*np.cos(2*math.pi*(AM+2)*3000*t_100)
# plt.vlines(t_100, [0], y, linewidth=0.3)
# plt.plot(t_100, y_100, '.', marker=".", markersize=4)
# plt.show()

##### version 3 #####
sign_sampling(100)

#III)
t_20 = np.linspace(0, N_periods*Tm, 80+1)
y_20 = A*np.cos(2*math.pi*Fm*t_20)*np.cos(2*math.pi*(AM+2)*Fm*t_20)
plt.vlines(t_20, [0], y_20, linewidth=0.8, colors="b")

t_100 = np.linspace(0, N_periods/Fm, 400+1)
y_100 = A*np.cos(2*math.pi*Fm*t_100)*np.cos(2*math.pi*(AM+2)*Fm*t_100)
plt.vlines(t_100, [0], y_100, linewidth=0.3)

plt.plot(t_20, y_20, '.', t_100, y_100, '.', 4/3000, y[0])
plt.legend(['fs1', 'fs2'])
plt.show()

#B ΕΡΏΤΗΜΑ
# t_5 = np.linspace(0,4/3000,20)
# y_5 = A*np.cos(2*math.pi*3000*t_5)*np.cos(2*math.pi*(AM+2)*3000*t_5)
# plt.vlines(t_5, [0], y_5, linewidth=0.3)
# plt.plot(t_5, y_5, '.', marker=".", markersize=4)
# plt.show()
sign_sampling(5)

#fourier spectrum
signal_fft = fftpack.fft(y)
Amplitude = np.abs(signal_fft)
Power = Amplitude**2
Angle = np.angle(signal_fft)
sample_freq = fftpack.fftfreq(y.size, d=Tm/8001)
Amp_freq = np.array([Amplitude, sample_freq])

plt.plot(abs(signal_fft), "o")
plt.show()

plt.subplot(2, 1, 1)
plt.plot(t, y)


t = np.linspace(0, N_periods*Tm, 8000+1)
y = A*np.cos(2*math.pi*Fm*t)*np.cos(2*math.pi*(AM+2)*Fm*t)
yf = fft(y)
tf = np.linspace(0.0, 1.0/(2.0*Tm), 8001//2)

plt.plot(tf, 2.0/8001 * np.abs(yf[0:8001//2]))
plt.grid()
plt.show()


from scipy.fft import fft
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = A*np.cos(2*math.pi*Fm*x)*np.cos(2*math.pi*(AM+2)*Fm*x)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

########### FINAL #######
#number of samples
N = 100*4
# sample spacing
T = 1.0 / (100.0*3000.0)
x = np.linspace(0.0, N*T, N)
y = np.cos(3000.0 * 2.0*np.pi*x)*np.cos(5.0*3000.0 * 2.0*np.pi*x)
plt.subplot(2, 1, 1)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) #for automatic x-scale (10^-3)
plt.plot(x, y)
yf = fft(y)
xf = fftpack.fftfreq(N, T)
xf = fftpack.fftshift(xf)
yplot = fftpack.fftshift(yf)
plt.subplot(2, 1, 2)
plt.xlim(0, 8*3000)
plt.xticks(np.arange(0, 11*3000, step=3000))
plt.plot(xf, 4.0/N * np.abs(yplot))
plt.grid()
plt.show()
