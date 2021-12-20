from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages                                   # to save pdf
from sklearn import linear_model                                                    # to use sklearn
#---------------------------------------------------------------------------------------------------
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T    # height (cm)
y = np.array([  49,  50,  51,  54,  58,  59,  60,  62,  63,  64,  66,  67,  68 ])      # weight (kg) 
# Visualize data 
plt.plot(X, y, 'ro')
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
# calculate weights of linear regression model------------------------------------------------------
# Building Xbar
one        = np.ones((X.shape[0], 1))                            # X.shape = (13,1) 
Xbar       = np.concatenate((one, X), axis = 1)                  # each row is one data point
A          = np.dot(Xbar.T, Xbar)                                # Xbar*(Xbar^T)
b          = np.dot(Xbar.T, y)                                   # Xbar*y
w          = np.dot(np.linalg.pinv(A), b)                        # pinv (pesudo inverse)
                                                                 # w =((Xbar*(Xbar^T))^(-1))*(Xbar*y)  
w0, w1     = w[0], w[1]                                          # weights
# linear regression model of SKLEARN ---------------------------------------------------------------
sk_lr      = linear_model.LinearRegression()
sk_lr.fit(X, y)                                        # in scikit-learn, each sample is one row
sk_w0      = sk_lr.intercept_
sk_w1      = sk_lr.coef_[0]
#---------------------------------------------------------------------------------------------------
x0         = np.linspace(145, 185, 2, endpoint=True)
y0         = w0    + w1*x0
y1         = sk_w0 + sk_w1*x0
# Drawing the fitting line 
plt.plot(X, y, 'ro')                                                         # data 
plt.plot(x0, y0)                                                             # the fitting line
plt.axis([140, 190, 45, 75])                                                 # xmin, xmax, ymin, ymax 
plt.xlabel('LR height (cm)', fontsize = 14)
plt.ylabel('LR weight (kg)', fontsize = 14)
plt.tick_params(axis='both', which='major', labelsize = 14)                  # to save pdf
with PdfPages('1_lr.pdf') as pdf:
    pdf.savefig(bbox_inches='tight')
plt.show()
# Drawing the fitting line 
plt.plot(X, y, 'ro')                                                         # data 
plt.plot(x0, y1)                                                             # the fitting line
plt.axis([140, 190, 45, 75])                                                 # xmin, xmax, ymin, ymax 
plt.xlabel('SK height (cm)', fontsize = 14)
plt.ylabel('SK weight (kg)', fontsize = 14)
plt.tick_params(axis='both', which='major', labelsize = 14)                  # to save pdf
with PdfPages('1_SK_lr.pdf') as pdf:
    pdf.savefig(bbox_inches='tight')
plt.show()
#---------------------------------------------------------------------------------------------------
weight_lr   = w1*155    + w0
print('Input 155cm, true output 52kg, predicted using LR output %.2fkg.' %(weight_lr))
weight_sk   = sk_w1*155 + sk_w0
print('Input 155cm, true output 52kg, predicted using SK output %.2fkg.' %(weight_sk))
#---------------------------------------------------------------------------------------------------