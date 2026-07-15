"""
TurboQuant-style Int8 quantization for the HN qwen3-8b embeddings.

Pipeline per vector x (dim d=4096=2^12):
  1. L2-normalize:            xn = x / ||x||
  2. randomized Hadamard:     r  = FWHT(signs * xn) / sqrt(d)     (orthonormal, seeded signs)
  3. standardize:             s  = r * sqrt(d)                    (components ~ N(0,1))
  4. scalar-quantize s with a 256-level Gaussian Lloyd-Max codebook -> 8-bit cell index 0..255

The codebook is EMBEDDED: the top b bits of the 8-bit index are themselves a valid
2^b-level quantizer (cells are unions of 8-bit cells), so Int4/Int2/binary are extracted
by  idx >> (8-b)  and reconstructed with codebook_b.  binary (b=1) == sign.
"""
import numpy as np
from scipy.stats import norm

D = 4096
SEED = 0x5152  # 'QR'

# ----- deterministic randomized Hadamard rotation -----
def rotation_signs(d=D, seed=SEED):
    return np.random.default_rng(seed).integers(0, 2, size=d).astype(np.float32) * 2 - 1  # +-1

def fwht(a):
    """In-place unnormalized fast Walsh-Hadamard transform along axis 1. a: (N,d), d=2^k."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n, d = a.shape
    h = 1
    while h < d:
        a = a.reshape(n, d // (2 * h), 2, h)
        u = a[:, :, 0, :]
        v = a[:, :, 1, :]
        diff = u - v          # one temp; computed before u is overwritten
        u += v                # a[:,:,0] = u + v  (in place)
        v[...] = diff         # a[:,:,1] = u_old - v
        a = a.reshape(n, d)
        h *= 2
    return a

def rotate(xn, signs):
    """xn: (N,d) already L2-normalized -> standardized rotated components ~N(0,1)."""
    r = fwht(xn * signs) / np.sqrt(xn.shape[1])  # orthonormal -> still unit norm
    return r * np.sqrt(xn.shape[1])              # ~N(0,1) per component

# ----- Gaussian Lloyd-Max + embedded codebooks -----
def lloyd_max_gaussian(nlev=256, iters=300):
    levels = norm.ppf((np.arange(nlev) + 0.5) / nlev)
    for _ in range(iters):
        bnd = (levels[:-1] + levels[1:]) / 2.0
        edges = np.concatenate(([-np.inf], bnd, [np.inf]))
        phi, Phi = norm.pdf(edges), norm.cdf(edges)
        new = (phi[:-1] - phi[1:]) / (Phi[1:] - Phi[:-1])
        if np.allclose(new, levels, atol=1e-12):
            levels = new; break
        levels = new
    bnd = (levels[:-1] + levels[1:]) / 2.0
    edges = np.concatenate(([-np.inf], bnd, [np.inf]))     # 257 edges
    return levels, edges

def embedded_codebook(edges8, b):
    """2^b reconstruction levels for the top-b-bit quantizer (cells = unions of 8-bit cells)."""
    step = 256 >> b
    ge = edges8[::step]                                     # 2^b + 1 edges
    phi, Phi = norm.pdf(ge), norm.cdf(ge)
    return (phi[:-1] - phi[1:]) / (Phi[1:] - Phi[:-1])

LEVELS8, EDGES8 = lloyd_max_gaussian(256)
CODEBOOKS = {b: embedded_codebook(EDGES8, b) for b in (1, 2, 4, 8)}

# ----- quantize / dequantize -----
def quantize(x, signs):
    """x: (N,d) raw embeddings -> uint8 cell indices 0..255 (np.uint8)."""
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)
    s = rotate(xn.astype(np.float32), signs)
    idx = np.searchsorted(EDGES8[1:-1], s).astype(np.uint8)  # 0..255
    return idx

def dequant(idx, b):
    """idx: uint8 8-bit indices -> reconstructed standardized vectors at b-bit precision."""
    g = (idx.astype(np.uint16) >> (8 - b))
    return CODEBOOKS[b][g].astype(np.float32)
