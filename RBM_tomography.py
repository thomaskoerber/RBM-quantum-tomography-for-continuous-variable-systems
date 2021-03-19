###########  import packages ###########
import numpy as np
import os 
import scipy.special as sc
import scipy.optimize as sci
import qutip as qt
import matplotlib.pyplot as plt
import random


########### define measurement and RBM model ###########

##################################################################################
######################### auxiliary functions ####################################
##################################################################################

def get_decimal(v):
    
    ## takes a binary vector v and returns its decimal representaion
    
    d = v.shape[0]
    return(np.matmul(2**np.arange(d),v))

def get_units(N):
    
    ## takes an integer N and returns an (N,2**N) array containing the binary representations of all ordered non-negative integers i < 2**n
    
    V = np.arange(0,2**N)%2
    for i in range(N-1):
        V = np.vstack((V,np.array(np.arange(0,2**N)% 2**(i+2)>(2**(i+1)-1)).astype(int)))
    return(V)

def coef(k):
    
    ## takes a non-negative integer k and returns an array c of length (k+1) with zero entries apart from c[k] = 1 
    
    c = np.zeros(k + 1)
    c[k] = 1
    return (c)

def normalized_hermite_polynomial(n, x):
    
    ## takes an integer n and a real number x and returns the L2-normalized physicist's Hermite polynomial H_n(x)
    
    c = (2 / np.pi) ** (1/4)
    
    H = np.polynomial.hermite.Hermite(coef(n[0]))(np.sqrt(2) * x).T
    for i in n[1:]:
        H = np.vstack((H,(np.polynomial.hermite.Hermite(coef(i))(np.sqrt(2) * x)).T))
        
    return(c * H * np.squeeze(np.tensordot(1 /  np.sqrt(2**n * sc.factorial(n)), np.exp(-x**2),axes=0)))


##################################################################################
##################### measurement-specific overlap function ######################
##################################################################################

## The method can be applied to different experimental setups. We model the setup by possible measurement
## settings theta and observed measurement results x. The dependence of the probability to observe the result x
## given the measurement setting theta and the system in the Fock state | n ⟩ depends of the specific measurement 
## model. To train the RBM, we need to know the overlap ⟨ theta, x | n ⟩ between | n ⟩ and the eigenstate 
## | theta , x ⟩ of the measurement corresponding to theta and x. We encode this information in the function 
##          specific_overlap(theta, x, v_0, v_1) = ⟨ m | theta, x ⟩⟨ theta, x | n ⟩,
## where v_0 (v_1) is the binary representation of n (m). 

## As an expemplary overlap, here we give the overlap function of a single-mode homodyne measurement (setting 
## theta is the relative phase of the local oscillator, result x is the homodyne current): 

def homodyne_overlap(theta,x,v_0,v_1): 
    
    ## takes homodyne measurement data theta (phase setting of measurement) and x (observed homodyne current), and visible units v_0, v_1 
    ## with decimal representations n, m, respectively, and computes the homodyne overlap
    ##  (H_n(x_j)*H_m(x_j)*exp(i theta (m-n))_{nmj}), where H_j is the L^2-normalized physicist's Hermite polynomial
    
    n = get_decimal(v_0)
    m = get_decimal(v_1)
    
    x = np.squeeze(x)
    theta = np.squeeze(theta)
    
    H_n = normalized_hermite_polynomial(n,x)
    H_m = normalized_hermite_polynomial(m,x)    
    H = np.diagonal(np.moveaxis(np.tensordot(H_n,H_m,axes=0),(0,1,2,3),(0,2,1,3)),0,2,3)

    diff = np.tensordot(np.ones(n.shape[0]),m,axes=0) -  np.tensordot(n,np.ones(m.shape[0]),axes=0) 
    
    return(np.exp(np.array(1j) * np.tensordot(diff,theta,axes=0)) * H)


##################################################################################
################################### model ########################################
##################################################################################

# model parameters

n_v = 2 # number of visible units
n_ha = 2 # number of hidden units, amplitude estimation
n_hp = 2 # number of hidden units, phase estimation 


# initialize units

V = get_units(n_v) # visible units
H_a = get_units(n_ha) # hidden units for amplitutde estimation
H_p = get_units(n_hp) # hidden units for phase estimation

# initialize weights

def initialize_parameters(n_v, n_ha, n_hp, sig=1.):
    
    ## takes a number of visible units n_v, a number of hidden units for amplitude estimation n_ha,
    ## a number of hidden units for phase estimation n_hp and a value sig and returns dictionaries
    ##  containing the intiaizlied biases and weights of the RBMs for amplitude and phase estimation,
    ##  respectively, with normally distributed values with mean 0 and standard deviation sig
    ## the subscripts 0,1 correspond neurons describing system and purification Hilbert space, respectively 
    
    ## amplitude estimation 
    
    A_a0 = np.random.normal(0,sig,(n_v,1)) 
    A_a1 = np.random.normal(0,sig,(n_v,1)) 
    B_a = np.random.normal(0,sig,(n_ha,1)) 
    W_a0 = np.random.normal(0,sig,(n_v,n_ha))
    W_a1 = np.random.normal(0,sig,(n_v,n_ha))

    ## phase estimation
    
    A_p0 = np.random.normal(0,sig,(n_v,1)) 
    A_p1 = np.random.normal(0,sig,(n_v,1)) 
    B_p = np.random.normal(0,sig,(n_hp,1)) 
    W_p0 = np.random.normal(0,sig,(n_v,n_hp))
    W_p1 = np.random.normal(0,sig,(n_v,n_hp))
    
    weights_amplitude = {"A_0": A_a0,
                         "A_1": A_a1,
                         "B": B_a,
                         "W_0": W_a0,
                         "W_1": W_a1}
    weights_phase = {"A_0": A_p0,
                     "A_1": A_p1,
                     "B": B_p,
                     "W_0": W_p0,
                     "W_1": W_p1}
    
    parameters = [weights_amplitude, weights_phase]
    
    return(parameters)

#model functions

def energy(v_0, v_1, h, weights): 
    
    ## takes visible units v_0, v_1, hidden units h, and weights (for amplitude estimation or phase estimation)
    ## and computes the matrix exp(-E(v_0, h)-E(v_1, h))_{v_0, v_1, h}, where
    ## E(v_i) =  - v_i^T W_i h + a_i^T v_i + b^T h
    
    W_0 = weights["W_0"]
    W_1 = weights["W_1"]
    A_0 = np.squeeze(weights["A_0"])
    A_1 = np.squeeze(weights["A_1"])
    B   = np.squeeze(weights["B"])
    
    ones_v_0 = np.ones(v_0.shape[1])
    ones_v_1 = np.ones(v_1.shape[1])
    ones_h = np.ones(h.shape[1])
     
    return(np.exp(  np.moveaxis(np.tensordot(ones_v_1,np.matmul(np.matmul(np.transpose(v_0),W_0),h),axes=0),(0,1,2),(1,0,2))
                  + np.tensordot(ones_v_0,np.matmul(np.matmul(np.transpose(v_1),W_1),h),axes=0)
                  + np.moveaxis(np.tensordot(ones_v_1,np.tensordot(np.matmul(A_0.T,v_0).T,ones_h,axes=0),axes=0),(0,1,2),(1,0,2))
                  + np.tensordot(ones_v_0,np.tensordot(np.matmul(A_1.T,v_1).T,ones_h,axes=0),axes=0)
                  + np.tensordot(np.tensordot(ones_v_0,ones_v_1,axes=0),np.matmul(np.transpose(B),h),axes=0)))

def energy_derivative(v_0, v_1, h, weights, key): 
    
    ## takes visible units v_0, v_1, hidden units h, weights (for amplitude estimation or phase estimation)
    ## and a key = W_0, W_1, A_0, A_1, B and returns (D_{key} exp(-E(v_0, h)-E(v_1, h)))_{v_0,v_1,h}
    
    E = energy(v_0, v_1, h, weights)
    
    if key == "W_0":
        ones_v_1 = np.ones(v_1.shape[1])
        return(np.moveaxis(np.tensordot(ones_v_1,np.tensordot(v_0,h,axes=0),axes=0),(0,1,2,3,4),(3,0,2,1,4))*E)
    
    elif key == "W_1":
        ones_v_0 = np.ones(v_0.shape[1])
        return(np.moveaxis(np.tensordot(ones_v_0,np.tensordot(v_1,h,axes=0),axes=0),(0,1,2,3,4),(2,0,3,1,4))*E)
    
    elif key == "A_0":
        ones_v_1 = np.ones(v_1.shape[1])
        ones_h = np.ones(h.shape[1])
        return(np.moveaxis(np.tensordot(np.tensordot(np.tensordot(v_0,ones_v_1,axes=0),ones_h,axes=0),np.ones(1),axes=0),(0,1,2,3,4),(0,2,3,4,1))*E)

    elif key == "A_1":
        ones_v_0 = np.ones(v_0.shape[1])
        ones_h = np.ones(h.shape[1])
        return(np.moveaxis(np.tensordot(np.tensordot(np.tensordot(v_1,ones_v_0,axes=0),ones_h,axes=0),np.ones(1),axes=0),(0,1,2,3,4),(0,3,2,4,1))*E)

    elif key == "B":
        ones_v_0 = np.ones(v_0.shape[1])
        ones_v_1 = np.ones(v_1.shape[1])
        return(np.moveaxis(np.tensordot(np.tensordot(np.tensordot(h,ones_v_0,axes=0),ones_v_1,axes=0),np.ones(1),axes=0),(0,1,2,3,4),(0,4,2,3,1))*E)
    
def measurement(theta, x, v_0, v_1, v_2, parameters, overlap): 
    
    ## takes general measurement settings theta, general measurement outcomes x, visible units v_0, v_1, v_2 with decimal representations n, m, k, respectively,
    ## the model parameters and an measurement-specific overlap function (e.g., the homodyne overlap function) and computes the tensor 
    ## (overlap(theta_j,x_j,n,m)*Sqrt(P_{nk}*P_{mk})*Exp( i*(phi_{nk}-phi_{mk})/2))_{mnkj},
    ## where P_{nm} = tr_h(energy_a(n, m, h)) and phi_{nm} = log(tr_h(energy_p(n, m, h)))
    
    weights_a = parameters[0]
    weights_p = parameters[1]
    
    P_nk = np.tensordot(energy(v_0, v_2, H_a, weights_a),np.ones(2**n_ha),axes=1)
    P_mk = np.tensordot(energy(v_1, v_2, H_a, weights_a),np.ones(2**n_ha),axes=1) 
    P = np.sqrt(np.tensordot(np.diagonal(np.moveaxis(np.tensordot(P_nk, P_mk, axes=0),(0,1,2,3),(0,2,1,3)),0,2,3),np.ones(x.shape[0]),axes=0))

    phi_nk =  np.log(np.tensordot(energy(v_0, v_2, H_p, weights_p),np.ones(2**n_hp), axes=1)) 
    phi_mk =  np.log(np.tensordot(energy(v_1, v_2, H_p, weights_p),np.ones(2**n_hp), axes=1)) 
    phi = np.tensordot((np.moveaxis(np.tensordot(np.ones(v_1.shape[1]),phi_nk,axes=0),(0,1,2),(1,0,2))-np.tensordot(np.ones(v_0.shape[1]),phi_mk,axes=0))/2,np.ones(x.shape[0]),axes=0)
    
    O = np.moveaxis(np.tensordot(np.ones(v_2.shape[1]),overlap(theta, x, v_0, v_1),axes=0),(0,1,2,3),(2,0,1,3)) 
    
    return(O * P * np.exp(np.array(1j) * phi))

def gradients(theta, x, parameters, overlap): 
    
    ## takes measurement settings theta and results x, model parameters, and an overlap function and returns a list 
    ## containing the gradients log-likelihood function with respect to amplitude and phase estimation and  
    ## the parameters A_0, A_1, B, W_0, W_1
    
    l = x.shape[0]
    Q = np.average(measurement(theta, x, V, V, V, parameters, overlap), axis=1)    

    gradients_amplitude = {"DA_0" : None, "DA_1" : None, "DB" : None, "DW_0" : None,"DW_1" : None }
    gradients_phase = {"DA_0" : None, "DA_1" : None, "DB" : None, "DW_0" : None,"DW_1" : None }
    
    grads = [gradients_amplitude, gradients_phase]

    for key_0 in {"a", "p"}:
        
        if key_0 == "a":
            
            H = H_a
            n_h = n_ha
            weights = parameters[0]
            
            E_neg = np.average(energy(V, V, H, weights))
            E_pos = np.tensordot(np.ones((1,1)),np.average(energy(V, V, H, weights), axis=2),axes=0)
            
            for key_1 in {"A_0", "A_1", "B", "W_0", "W_1"}:
                
                D_neg = np.average(energy_derivative(V, V, H, weights, key_1),axis=(2,3,4))
                D_pos = np.average(energy_derivative(V, V, H, weights, key_1), axis=4)

                gradients_amplitude["D" + key_1] = - D_neg/E_neg + 1 / l * np.real(np.tensordot(np.tensordot(D_pos/E_pos,Q,axes=((2,3),(0,1))),1 / np.sum(Q, axis=(0,1)),axes=(2,0)))

        if key_0 == "p":
            H = H_p
            n_h = n_hp
            weights = parameters[1]
            
            E_pos = np.tensordot(np.ones((1,1)),np.average(energy(V, V, H, weights), axis=2),axes=0)
            
            for key_1 in {"A_0", "A_1", "B", "W_0", "W_1"}:
                
                D_pos = np.average(energy_derivative(V, V, H, weights, key_1), axis=4)
                
                gradients_phase["D" + key_1] = - 1 / l * np.imag(np.tensordot(np.tensordot(D_pos/E_pos,Q,axes=((2,3),(0,1))),1 / np.sum(Q, axis=(0,1)),axes=(2,0)))
    
    return(grads)

    
def log_likelihood(theta, x, parameters, overlap, splits = False, nsplits = 1): 
    
    ## takes measurement settings theta and results x, model parameters, and an overlap function and computes the log-likelihood 
    ## of the observed data given the model, with the option to split the computation into nsplits blocks
    
    
    weights_a = parameters[0]
    weights_p = parameters[1]
    
    E_a = energy(V, V, H_a, weights_a)
    Z_a = np.tensordot(E_a,np.ones((2**n_v, 2**n_v, 2**n_ha)), axes=3)
    
    if splits == False: 
        
        l = x.shape[0]
        Q = measurement(theta, x, V, V, V, parameters, overlap)
        llk = -l * np.log(Z_a) + np.tensordot(np.log(np.absolute(np.tensordot(Q,np.ones((2**n_v,2**n_v,2**n_v)),axes=[(0,1,2),(0,1,2)]))),np.ones(l),axes=1)
    if splits == True:
        
        x_split = np.array_split(x,nsplits)
        theta_split = np.array_split(theta,nsplits)
    
        llk = 0.
        for i in range(nsplits):
            l = x_split[i].shape[0]
            Q = measurement(theta_split[i], x_split[i], V, V, V, parameters, overlap)
            llk += -l * np.log(Z_a) + np.tensordot(np.log(np.absolute(np.tensordot(Q,np.ones((2**n_v,2**n_v,2**n_v)),axes=[(0,1,2),(0,1,2)]))),np.ones(l),axes=1)
    
    return(llk) 

def reconstruct_from_RBM(parameters):
    
    ## takes model parameters and computes the corresponding estimation of the density matrix of the quantum state
    
    weights_a, weights_p = parameters
    
    Z_a = np.tensordot(energy(V, V, H_a, weights_a),np.ones((2 ** n_v,2 ** n_v,2 ** n_ha)),axes=3)
    
    P_nk = np.tensordot(energy(V, V, H_a, weights_a),np.ones(2 ** n_ha),axes=1) 
    P_mk = np.tensordot(energy(V, V, H_a, weights_a),np.ones(2 ** n_ha),axes=1) 
    
    phi_nk = np.log(np.tensordot(energy(V, V, H_p, weights_p),np.ones(2 ** n_hp),axes=1)) 
    phi_mk = np.log(np.tensordot(energy(V, V, H_p, weights_p),np.ones(2 ** n_hp),axes=1)) 
    
    return(np.tensordot(np.diagonal(np.sqrt(np.tensordot(P_nk,P_mk,axes=0)) * np.exp(np.array(1j * .5)*(np.tensordot(phi_nk,np.ones((2 ** n_v,2 ** n_v)),axes=0)
    	                -np.tensordot(np.ones((2 ** n_v,2 ** n_v)),phi_mk,axes=0))),0,1,3),np.ones(2 ** n_v),axes=1) / Z_a)

#training the model

def update_parameters(parameters, gradients, learning_rate = 0.01):
    
    ## takes model parameters, gradients, and a learning rate and updates the model parameters according to
    ## gradient ascent
    
    weights_a, weights_p = parameters
    grads_a, grads_p = gradients
    
    for key in weights_a:
        weights_a[key] = weights_a[key] + learning_rate * grads_a["D" + key]
        weights_p[key] = weights_p[key] + learning_rate * grads_p["D" + key]
        
    return([weights_a, weights_p])

def train_model(theta, x, parameters, overlap, batchsize = 50, learning_rate = 0.1, epochs = 5000,  verbose = True, nsplits = 50):

    ## takes measurement settings theta and results x, model parameters, an overlap function, a batchsize, a learning_rate, a number of epochs, 
    ## a boolean parameter specifying if the log-likelihood will be computed and printed during gradient ascent, 
    ## and a number of splits for the log likelihood computation and returns the parameters of the model trained
    ## by stochastic gradient ascent
    
    assert isinstance(theta,np.ndarray), "theta is not a numpy array"
    
    assert isinstance(x,np.ndarray), "x is not a numpy array"

    assert x.shape[0] == theta.shape[0], "theta and x have different sizes"
    
    assert batchsize < x.shape[0], "the batchsize is larger than the size of x"
    
    assert 2 * nsplits < x.shape[0], "the number of splits is too large"

   
    for j in range(epochs):
        batch = random.sample(range(0, x.shape[0]-1), batchsize)
        x_batch = x[batch]
        theta_batch = theta[batch]
                
        grads = gradients(theta_batch, x_batch, parameters, overlap)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        
        
        if j%500==0 and verbose == True:
            print('Epoch:',j,'Loglikelihood:',log_likelihood(theta, x, parameters, overlap, splits = True, nsplits = nsplits))
    return(parameters)




#### example: homodyne measurement for single-photon-added coherent states ###########

# define fidely between states of possibly different cutoff dimensions

def fidel_diffdims(A,B ):
    
    ## takes to density matrices A,B and computes the quantum fidelity (also if A and B have different cutoffs), using the qutip package
    
    len1,len2=len(A),len(B)

    if len1 < len2:
        a = np.zeros(B.shape,dtype=complex)
        a[:A.shape[0],:A.shape[1]] = A
        A=a
    
    elif len1 > len2:
        b = np.zeros(A.shape,dtype=complex)
        b[:B.shape[0],:B.shape[1]] = B
        B=b
    
    A, B = qt.Qobj(A), qt.Qobj(B)
    
    return (qt.metrics.fidelity(A,B))**2

# quadrature probabilities for SPACS 

def PSpacs(theta, x, alp, eta):
    
    ## probability density to observe homodyne current x given a measurement of a SPACS with amplitude alp
    ## using a measurement setting theta and detection efficiency eta
    ## source: Eq. 18 in A. Zavatta et al., Phys. Rev. A 72, 023820 (2005)
    
    return np.real(1 / (np.sqrt(np.pi * 2 * .25) * (1 + np.absolute(alp) ** 2)) * ( eta * (2 * x * np.cos(theta - np.angle(alp)) - (2. * eta- 1.) * alp / np.sqrt(eta)) ** 2 
                   + 4 * eta * x ** 2 * np.sin(theta - np.angle(alp)) ** 2 + (1. - eta) * (1. + 4 * eta * np.absolute(alp) ** 2 * np.sin(theta - np.angle(alp)) ** 2) ) 
                   * np.exp(-0.5 / .25 * (x - np.sqrt(eta) * np.absolute(alp) * np.cos(theta-np.angle(alp))) ** 2))

# simulate quadrature data for SPACS

def ZSpacs(theta, grid, alp, eta, n_samp):
    
    ## generates n_samp random results according to PSpacs(theta,x,alp,eta), corsegrained on the grid grid
    
    return np.random.choice(grid,n_samp,p = PSpacs(theta, grid, alp, eta)/sum(PSpacs(theta, grid, alp, eta)))  

# auxilary function to include noise in theoretical prediction

def B(m, k, eta):
    
    ## auxilary function to include the effect of finite detection efficiency in homodyne detection
    ## source: Eq. 21 in A. Zavatta et al., Phys. Rev. A 72, 023820 (2005)
    
    return(np.sqrt(eta ** m * (1. - eta) ** k * (sc.factorial(m+k) / sc.factorial(m) / sc.factorial(k))))

# theoretical prediction of SPACS density matrix

def SPACS_theory(alp, eta, d, d_k): # generates noisy density matrix from a SPACS with inital coherent amplitude alp, det efficiency eta and cutoff dimension d, cut_off of k sum d_k
    
    ## computes the theoretical prediction of a density matrix for a SPACS with amplitude alp including the measurement
    ## efficiency eta, using a cutoff dimension d for the systems and d_k for the noise correction
    ## source: Eq. 3 and Eq. 21 in A. Zavatta et al., Phys. Rev. A 72, 023820 (2005)
    
    m = np.array(range(d + d_k))
    n = np.array(range(d + d_k))
    m_rho_n = np.tensordot(np.sqrt(m) * np.concatenate([np.array([1.]),alp ** (m[1:] - 1) / np.sqrt(sc.factorial(m[1:] - 1))]),np.sqrt(n) * np.concatenate([np.array([1.]),np.conjugate(alp ** (n[1:] - 1)) / np.sqrt(sc.factorial(n[1:] - 1))]),axes=0)   * (np.exp(-np.absolute(alp) ** 2) / (1 + np.absolute(alp) ** 2))
    
    out = np.full((d,d),0. + 1j * 0.)
    for k in range(d_k):
        out += np.tensordot(B(m[:d],k,eta),B(n[:d],k,eta),axes=0) * m_rho_n[k:k + d,k:k + d]
    
    return(out)


# generate training data 
alpha, eta  = 0.2 * np.exp(1j * np.pi/2), .6
grid_SPACS = np.linspace(-4., 4., 1000) # grid for sampling of homodyne data
N_meas = 3000 # number of measurements
theta_0 = np.random.random(N_meas) * 2 * np.pi # measurement settings
x_0 = [] # measurement results
for k in range(N_meas):
    x_0.append(ZSpacs(theta_0[k], grid_SPACS, alpha, eta, 1))
x_0=np.array(x_0).flatten()

# initialize weights of RBM
parameters_0 = initialize_parameters(n_v=2, n_ha=2, n_hp=2, sig=1.)
# train RBM with simualted data
parameters_1 = train_model(theta_0, x_0 , parameters_0, homodyne_overlap, batchsize = 100, learning_rate = 0.5, epochs = 20000, verbose = True, nsplits = 10)

# fidelity between reconstructed and theoretically predicted state
rho_reconstructed = reconstruct_from_RBM(parameters_1)
rho_predcition = SPACS_theory(alpha, eta, 4, 10)

print('Fidelity between reconstructed state and theoretical prediction for a SPACS: ')
print(fidel_diffdims(rho_reconstructed,rho_predcition))

