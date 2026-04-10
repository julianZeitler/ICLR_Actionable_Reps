# File containing all the various loss functions I have written
import jax.numpy as jnp
import jax
from nrt import helpers

### LOSSES FOR CIRCLE, 3 different measures of separation quality, 1 for positivity, and 1 for equivariance
### ALSO THE SAME LOSSES AS FOR TORUS, ALL JUST WORKS!

def sep_circ_EucChi(W, I, Chi):
    # Set up the normalised version of the weight matrix
    norms = jnp.linalg.norm(W,axis=1)
    W = W/norms[:,None]

    # Create the neural responses
    g = jnp.matmul(W, I)
    N = g.shape[1]

    # Measure the separation
    Xi = -jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2),axis=0)
    L_Sep = 1/jnp.power(N,2)*jnp.sum(jnp.multiply(Xi, Chi))
    return L_Sep

def sep_circ_Euc(W, I):
    # Set up the normalised version of the weight matrix
    norms = jnp.linalg.norm(W,axis=1)
    W = W/norms[:,None]

    # Create the neural responses
    g = jnp.matmul(W, I)
    N = g.shape[1]

    # Measure the separation
    Xi = -jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2),axis=0)
    L_Sep = 1/jnp.power(N,2)*jnp.sum(Xi)
    return L_Sep

def sep_circ_Kern(W, I, sigma_sq):
    # Set up the normalised version of the weight matrix
    norms = jnp.linalg.norm(W,axis=1)
    W = W/norms[:,None]

    # Create the neural responses
    g = jnp.matmul(W, I)
    N = g.shape[1]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2)/(2*sigma_sq),axis=0)) # the guassian bump
    return jnp.sum(Xi)/jnp.power(N,2)
def sep_circ_KernChi(W, I, sigma_sq, Chi):
    # Set up the normalised version of the weight matrix
    norms = jnp.linalg.norm(W,axis=1)
    W = W/norms[:,None]

    # Create the neural responses
    g = jnp.matmul(W, I)
    N = g.shape[1]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2)/(2*sigma_sq),axis=0)) # the guassian bump
    L_Sep = 1/jnp.power(N,2)*jnp.sum(jnp.multiply(Xi, Chi))
    return L_Sep

# Positivity loss linear
def pos_circ(W, I):
    # Set up the normalised version of the weight matrix
    norms = jnp.linalg.norm(W,axis=1)
    W = W/norms[:,None]

    # Create the neural responses
    V = jnp.matmul(W, I)
    [D, N] = V.shape

    # Ignore all the positive entries
    V_Neg = (V - jnp.abs(V))/2

    # Then measure the positivity
    L_pos = -jnp.sum(V_Neg)/(D*N)
    return L_pos

# Same as above but learnable return matrix
def equi_circ_smart_B(W, B, I_base, I_shift, G_I):
    # Set up the normalised version of the weight matrix
    norms = jnp.linalg.norm(W,axis=1)
    W = W/norms[:,None]

    V_base = jnp.matmul(W,I_base)
    V_shift = jnp.matmul(W,I_shift)
    G = jnp.einsum('ij,kjp->kip', W, jnp.einsum('ikl,lp->ikp', G_I, B))
    V_tilde = jnp.einsum('kij,jl->ilk',G, V_base)
    V_tilde_reshape = jnp.reshape(V_tilde, [V_tilde.shape[0], V_tilde.shape[1]*V_tilde.shape[2]],order='F')
    L = jnp.sum(jnp.power(V_tilde_reshape - V_shift,2))/(V_tilde_reshape.shape[1])
    return L

def equi_smart(W, I_base, I_shift, G_I):
    # Set up the normalised version of the weight matrix
    norms = jnp.linalg.norm(W,axis=1)
    W = W/norms[:,None]

    V_base = jnp.matmul(W,I_base)
    V_shift = jnp.matmul(W,I_shift)
    G = jnp.einsum('ij,kjp->kip', W, jnp.einsum('ikl,lp->ikp', G_I, jnp.linalg.pinv(W)))
    V_tilde = jnp.einsum('kij,jl->ilk',G, V_base)
    V_tilde_reshape = jnp.reshape(V_tilde, [V_tilde.shape[0], V_tilde.shape[1]*V_tilde.shape[2]],order='F')
    L = jnp.sum(jnp.power(V_tilde_reshape - V_shift,2))/(V_tilde_reshape.shape[1])
    return L

### LOSSES FOR LINE 3 different separation, positivity, and a norm one

def pos_line(W, om, phi, N_shift):
    # Create the irrep basis
    I = helpers.init_irreps_1D(om, phi)

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)/(N_shift+1)
    g = g/norms[:,None]
    [D, N] = g.shape

    # measure positivity
    g_neg = (g - jnp.abs(g))/2
    return -jnp.sum(g_neg)/(D*N)

def sep_line_Euc(W, om, phi):
    # Create the irrep basis
    I = helpers.init_irreps_1D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure positivity
    Xi = -jnp.sum(jnp.power(g[:, None, :] - g[:, :, None], 2), axis=0)
    return jnp.sum(Xi)/jnp.power(N,1)

def sep_line_Kern(W, om, phi, sigma_sq):
    # Create the irrep basis
    I = helpers.init_irreps_1D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2)/(2*sigma_sq),axis=0)) # the guassian bump
    return jnp.sum(Xi)/jnp.power(N,2)

def sep_line_EucChi(W, om, phi, chi):
    # Create the irrep basis
    I = helpers.init_irreps_1D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure separation
    Xi = -jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2),axis=0)
    return 1/jnp.power(N,2)*jnp.sum(jnp.multiply(Xi, chi))

def sep_line_KernChi(W, om, phi, sigma_sq, chi):
    # Create the irrep basis
    I = helpers.init_irreps_1D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2)/(2*sigma_sq),axis=0)) # the guassian bump
    return jnp.sum(jnp.multiply(Xi, chi))/jnp.power(N,3/2)

def norm_line(W, om, phi_room, phi_other):
    # Create the irrep basis
    I_room = helpers.init_irreps_1D(om, phi_room)
    I_other = helpers.init_irreps_1D(om, phi_other)

    # Use the room to normalise the other representations
    g_room = jnp.matmul(W, I_room)
    g_other = jnp.matmul(W, I_other)
    norms = jnp.linalg.norm(g_room, axis = 1)
    g = g_other/norms[:,None]
    [D, N] = g_room.shape
    N_shift = int(phi_other.size/N)

    # Measure the resulting norms in each of the rooms and penlise deviations from 1
    norms = jnp.sum(jnp.reshape(jnp.power(g_other, 2), [D, N_shift, N]), axis = 2)
    return jnp.linalg.norm(norms - 1)/(D*N_shift)

### LOSSES FOR PLANE 3 different separation, positivity, and a norm one
# SAME AS FOR LINE, JUST USING 2D IRREP INIT!

def pos_plane(W, om, phi, N_shift):
    # Create the irrep basis
    I = helpers.init_irreps_2D(om, phi)

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)/(N_shift+1)
    g = g/norms[:,None]
    [D, N] = g.shape

    # measure positivity
    g_neg = (g - jnp.abs(g))/2
    return -jnp.sum(g_neg)/(D*N)

def pos_plane_seq(g0, om, S, phi):
    # g0 = activity at origin (shape [D, 1])
    # T(phi) = S @ T_irrep(phi) @ S^(-1)
    # g(phi) = T(phi) @ g0
    B, L = phi.shape[0], phi.shape[1]

    # Apply softplus to ensure non-negativity and normalize
    g0 = jax.nn.softplus(g0)
    g0 = g0 / jnp.linalg.norm(g0)

    # Get transformation matrices for all phi positions
    # T shape: [B, L, D, D]
    shift_phi = jnp.roll(phi, 1, axis=1)
    shift_phi = shift_phi.at[:,0,:].set(0)
    d_phi = phi - shift_phi
    T = helpers.get_T_2D(om, d_phi, S)

    g = helpers.calc_g(g0, T) # g shape: [B, L, D]

    # measure positivity
    g_neg = (g - jnp.abs(g))/2
    return -jnp.sum(g_neg)/(B*L*g.shape[2])

def sep_plane_Euc(W, om, phi):
    # Create the irrep basis
    I = helpers.init_irreps_2D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure positivity
    Xi = -jnp.sum(jnp.power(g[:, None, :] - g[:, :, None], 2), axis=0)
    return jnp.sum(Xi)/jnp.power(N,2)

def sep_plane_Kern(W, om, phi, sigma_sq):
    # Create the irrep basis
    I = helpers.init_irreps_2D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2)/(2*sigma_sq),axis=0)) # the guassian bump
    return jnp.sum(Xi)/jnp.power(N,2)

def sep_plane_EucChi(W, om, phi, chi):
    # Create the irrep basis
    I = helpers.init_irreps_2D(om, phi)
    N = phi.shape[0]

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure separation
    Xi = -jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2),axis=0)
    return 1/jnp.power(N,2)*jnp.sum(jnp.multiply(Xi, chi))

def sep_plane_KernChi(W, om, phi, sigma_sq, chi):
    # Create the irrep basis
    I = helpers.init_irreps_2D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis = 1)
    g = g/norms[:,None]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2)/(2*sigma_sq),axis=0)) # the guassian bump
    return jnp.sum(jnp.multiply(Xi, chi))/jnp.power(N,2)

def sep_plane_KernChi_seq(g0, om, S, phi, sigma_sq, chi):
    # g0 = activity at origin (shape [D, 1])
    # T(phi) = S @ T_irrep(phi) @ S^(-1)
    # g(phi) = T(phi) @ g

    B, L = phi.shape[0], phi.shape[1]

    # Apply softplus to ensure non-negativity and normalize
    g0 = jax.nn.softplus(g0)
    g0 = g0 / jnp.linalg.norm(g0)

    # Get transformation matrices for all phi positions
    # T shape: [B, L, D, D]
    shift_phi = jnp.roll(phi, 1, axis=1)
    shift_phi = shift_phi.at[:,0,:].set(0)
    d_phi = phi - shift_phi
    T = helpers.get_T_2D(om, d_phi, S)

    g = helpers.calc_g(g0, T) # g shape: [B, L, D]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:,:] - g[:,:,None,:],2)/(2*sigma_sq),axis=3)) # the guassian bump across L
    return jnp.sum(jnp.multiply(Xi, chi))/(B*L*L) # Mean

def sep_plane_KernChi_seq_causal(g0, om, S, phi, sigma_sq, chi, decay=None):
    # g0 = activity at origin (shape [D, 1])
    # T(phi) = S @ T_irrep(phi) @ S^(-1)
    # g(phi) = T(phi) @ g

    B, L = phi.shape[0], phi.shape[1]

    # Apply softplus to ensure non-negativity and normalize
    g0 = jax.nn.softplus(g0)
    g0 = g0 / jnp.linalg.norm(g0)

    # Get transformation matrices for all phi positions
    # T shape: [B, L, D, D]
    shift_phi = jnp.roll(phi, 1, axis=1)
    shift_phi = shift_phi.at[:,0,:].set(0)
    d_phi = phi - shift_phi
    T = helpers.get_T_2D(om, d_phi, S)

    g = helpers.calc_g(g0, T) # g shape: [B, L, D]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:,:] - g[:,:,None,:],2)/(2*sigma_sq),axis=3)) # the guassian bump across L
    Xi = jnp.tril(Xi) # Mask "future" data
    if decay:
        # array with "decay" exponential decay starting from 1, length L
        decay_arr = jnp.power(decay, jnp.arange(L))  # [1, d, d^2, ..., d^(L-1)]
        # Matrix with this decay array arranged vertically with every column shifting the array down once. Don't wrap around (top triangle zero)
        row_idx, col_idx = jnp.meshgrid(jnp.arange(L), jnp.arange(L), indexing='ij')
        decay_matrix = jnp.where(row_idx >= col_idx, decay_arr[row_idx - col_idx], 0.0)
        Xi = jnp.multiply(Xi, decay_matrix[None, :, :])
    return jnp.sum(jnp.multiply(Xi, chi))/(B*L*L) # Mean

def sep_plane_KernChi_Module(W, grid_params, phi, sigma_sq, chi):
    # Create the frequencies
    M = W.shape[0]
    M_Q = int(M/2)
    om_1 = helpers.freq_module_plane_new(grid_params[0:4], M_Q)
    om_2 = helpers.freq_module_place_new(grid_params[4:], M_Q)
    om = jnp.vstack([om_1, om_2])

    # Create the irrep basis
    I = helpers.init_irreps_2D(om, phi)
    N = phi.size

    # Turn into normalised neural activity
    g = jnp.matmul(W, I)
    norms = jnp.linalg.norm(g, axis=1)
    g = g / norms[:, None]

    # measure separation
    Xi = jnp.exp(-jnp.sum(jnp.power(g[:,None,:] - g[:,:,None],2)/(2*sigma_sq),axis=0)) # the guassian bump
    return jnp.sum(jnp.multiply(Xi, chi))/jnp.power(N,2)


def norm_plane(W, om, phi_room, phi_other):
    # Create the irrep basis
    I_room = helpers.init_irreps_2D(om, phi_room)
    I_other = helpers.init_irreps_2D(om, phi_other)

    # Use the room to normalise the other representations
    g_room = jnp.matmul(W, I_room)
    g_other = jnp.matmul(W, I_other)
    norms = jax.lax.stop_gradient(jnp.linalg.norm(g_room, axis = 1))
    g = g_other/norms[:,None]
    [D, N] = g_room.shape
    N_shift = int(phi_other.shape[0]/N)

    # Measure the resulting norms in each of the rooms and penlise deviations from 1
    norms = jnp.sum(jnp.reshape(jnp.power(g, 2), [D, N_shift, N]), axis = 2)
    return jnp.linalg.norm(norms - 1)/(D*N_shift)

def norm_plane_seq(g0, om, S, phi_room, phi_other):
    B,L = phi_room.shape[0], phi_room.shape[1]
    N_shift = int(phi_other.shape[0]/B)

    # Apply softplus to ensure non-negativity and normalize
    g0 = jax.nn.softplus(g0)
    g0 = g0 / jnp.linalg.norm(g0)

    # Get transformation matrices for all phi positions
    # T shape: [B, L, D, D]
    shift_phi_room = jnp.roll(phi_room, 1, axis=1)
    shift_phi_room = shift_phi_room.at[:,0,:].set(0)
    d_phi_room = phi_room - shift_phi_room
    T_room = helpers.get_T_2D(om, d_phi_room, S)

    shift_phi_other = jnp.roll(phi_other, 1, axis=1)
    shift_phi_other = shift_phi_other.at[:,0,:].set(0)
    d_phi_other = phi_other - shift_phi_other
    T_other = helpers.get_T_2D(om, d_phi_other, S)

    g_room = helpers.calc_g(g0, T_room, norm=False) # g shape: [B, L, D]
    g_other = helpers.calc_g(g0, T_other, norm=False) # g shape: [B, L, D]

    [_,_,D] = g_room.shape

    # Use the room to normalise the other representations
    # ASSUMPTION: norm can be calculated from B*L
    norms = jax.lax.stop_gradient(jnp.linalg.norm(g_room.reshape([B*L,D]), axis=0, keepdims=True))
    g = g_other / norms
    
    # Measure the resulting norms in each of the rooms and penalise deviations from 1
    norms = jnp.sum(jnp.reshape(jnp.power(g, 2), [B,N_shift,L,D]), axis = [0,2]) # Sum over all points in a "room"
    return jnp.linalg.norm(norms - 1)/(D*N_shift)

# Set of losses for the sphere

def sep_sphere_Euc(W, I):
    I = helpers.init_irreps_sphere()