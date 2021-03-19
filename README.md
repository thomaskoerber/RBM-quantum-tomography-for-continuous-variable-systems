# Quantum state tomography of continous varibale systems with restricted Boltzmann machines (RBM)
This code implements a RBM quanutm state tomography for general measurements of quantum systems: The RBM is trained to maximize the expected probability (loglikelihood) of the 
input measurement data. The code follows closely the RBM quantum state reconstruction methods described in E.S. Tiunov et al., Optica 7, 448 (2020). 
The code can be applied to general measurement data, as long as the measurement can be formally modeled by a measurement-specific input (overlap) function. 
In addition, since the training becomes very slow for larger RBM sizes (as already noted in Tiunov et al.), we also implement different approximation methods of 
the training gradient by (a) a random sampling of the different terms in the gradient's sums, and (b) Gibbs sampling of the different terms in the gradient's sums. 
We observe that both approximation methods do not well approximate the complete training gradient, such that the simplified training never reaches good likelihoods 
(and the reconstructed state does not reach high fidelities to the target state). Further approximation methods should be considered. 
As an example, we apply the method to simulated measurement data from single-mode homodyne tomography.

The description of files
-------------------------

1. RBM_tomography.py - a program in pyhton for performing the RBM quantum state reconstruction described in E.S. Tiunov et al., Optica 7, 448 (2020) for a general 
quantum measurmenet. In addition, the RBM reconstruction is conducted for simulated noisy homodyne measurement data of a single-photon-added coherent state.

2. RBM_tomography_sampling.py - a program in pyhton for performing the RBM quantum state reconstruction, with the option that the gradients of the RBM training 
are approximated by different methods: (a) a random sampling of the different terms in the gradient's sums, (b) Gibbs sampling of the different terms in the gradient's sums.


Authorship statement
-------------------------
Files 1, 2 have been produced jointly by Thomas Koerber and Valentin Gebhart. The code mainly implements methods described in 
E.S. Tiunov et al., Optica 7, 448 (2020).
