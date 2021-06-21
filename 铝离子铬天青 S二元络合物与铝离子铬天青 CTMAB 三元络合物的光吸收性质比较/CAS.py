import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
def runplt(size=None):
    plt.figure(figsize=(10,6))
    plt.title('Absorption curve',family = 'serif')
    plt.xlabel(r'$Wavelength$/nm',family = 'serif')
    plt.ylabel('$Absorbance$',family = 'serif')
    plt.axis([490,710,0,0.6])
    return plt

X = [500,510,520,530,540,545,550,555,560,570,580,590,600]
y = [0.016,0.043,0.079,0.139,0.208,0.241,0.243,0.220,0.190,0.111,0.083,0.076,0.066]

X1 = [550,560,570,580,590,600,610,620,625,630,635,640,650,660,670,680,690,700]
y1 = [0.119,0.151,0.201,0.261,0.322,0.389,0.469,0.538,0.574,0.567,0.567,0.522,0.384,0.241,0.117,0.061,0.025,0.011]


x11 = np.linspace(500,600,500)
y11 = make_interp_spline(X, y)(x11)

x22 = np.linspace(550,700,500)
y22 = make_interp_spline(X1, y1)(x22)
pltt = runplt()
pltt.grid(zorder=0)
pltt.plot(x11, y11,zorder=2,label = 'Binary complexes')
pltt.plot(x22, y22,zorder=3,label = 'Ternary complexes')
print(np.where(y11 > 0.246305))
print(np.where(y22 > 0.574935))
pltt.scatter(x11[238], max(y11),marker = '*',zorder=4)
pltt.scatter(X,y,c = 'C0',zorder=6)
pltt.scatter([550,560,570,580,590,600,610,620,630,635,640,650,660,670,680,690,700], [0.119,0.151,0.201,0.261,0.322,0.389,0.469,0.538,0.567,0.567,0.522,0.384,0.241,0.117,0.061,0.025,0.011],zorder=5)
pltt.scatter(x22[252],  max(y22),marker = '*',c = 'C1',zorder=5)
pltt.text(546, 0.25, (np.around(x11[238],1),np.around( max(y11),3)),ha='left', va='bottom', fontsize=10,family = 'serif')
pltt.text(623, 0.576, (np.around(x22[252],1),  np.around(max(y22),3)),ha='right', va='bottom', fontsize=10,family = 'serif')
pltt.legend(loc='upper right',)
from matplotlib import ticker
pltt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "%.3f" % x))
# ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: "[%.2f]" % x))

# plt.text(0.4009, 0.2441, (0.4009,0.2441),ha='left', va='bottom', fontsize=10)
pltt.savefig("CAS.pdf")
pltt.show()
