""" Evaluate the scattering amplitudes and scattering/extinction
cross-sections of a dielectric sphere using Lorenz-Mie theory.

Based on an implementation by Kate Salesin and Wenzel Jakob adapted from:

   "Lorenz-Mie FORTRAN Programs for Computing Far-Field Electromagnetic 
    Scattering by Monodisperse and Polydisperse Spherical Particles." 
    Mishchenko, Travis, and Lacis. Scattering, Absorption, and Emission of 
    Light by Small Particles. Cambridge University Press, Cambridge. 2002.

    https://www.giss.nasa.gov/staff/mmishchenko/books.html

Units of distance are arbitrary but need to be consistent between
``radius``, ``wavelength``, and the imaginary components of the provided
refractive indices.

- radius
   Radius of the sphere (units of distance)
- wavelength
   Wavelength of the incident and scattered light (units of distance)
- ior_med
   Index of refraction of the most medium (complex-valued, a negative
   imaginary component implies absorption)
- ior_sph
   Index of refraction of the sphere (complex-valued, a negative
   imaginary component implies absorption)
- mu
   Cosine of angle between the incident and scattered direction
- nmax
   Number of series terms that should be used (-1: automatic)

Returns:
   A tuple (S1, S2, Ns, Ct, Cs) where
   - S1: scattering amplitude/phase of the *ordinary* ray
   - S2: scattering amplitude/phase of the *extraordinary* ray
   - Ns: normalization coefficient required by the phase function,
     which is defined as ``(abs(S1)^2 + abs(S2)^2) / Ns``
   - Cs: scattering cross section (units of 1/(distance * wavelength))
   - Ct: extinction cross section (units of 1/(distance * wavelength))
"""
import numpy as np

def mie(radius, wavelength, ior_sph, ior_med, mu, nmax = -1):

    # Relative index of refraction
    m = ior_sph / ior_med

    # Wave numbers
    kx = 2. * np.pi * ior_med / wavelength
    ky = 2. * np.pi * ior_sph / wavelength

    # Size parameters
    x = kx * radius
    y = ky * radius

    # Related constants
    m_sq = m * m
    rcp_x = 1. / x
    rcp_y = 1. / y
    x_norm = np.abs(x)
    y_norm = np.abs(y)
    kx_norm = np.abs(kx)
    kx_sq_norm = kx_norm * kx_norm

    x_nmax = nmax
    y_nmax = nmax

    # Default stopping criterion, [Mishchenko and Yang 2018]
    if (x_nmax == -1):
        x_nmax = 8 + x_norm + 4.05 * np.power(x_norm, 1. / 3.)
        y_nmax = 8 + y_norm + 4.05 * np.power(y_norm, 1. / 3.)

    # Default starting n for downward recurrence of ratio j_n(z) / j_{n-1}(z)
    x_ndown = int(x_nmax + 8 * np.sqrt(x_nmax) + 3)
    y_ndown = int(y_nmax + 8 * np.sqrt(y_nmax) + 3)

    # Calculate ratio j_n(z) / j_{n-1}(z) by downward recurrence for z = x
    j_ratio_x = np.zeros(x_ndown, dtype = np.cdouble)
    j_ratio_x_n = x / (2. * x_ndown + 1)

    n = x_ndown - 1
    while n > 0:
        kx_n = (2 * n + 1) * rcp_x
        j_ratio_x[n] = j_ratio_x_n = 1. / (kx_n - j_ratio_x_n)
        n -= 1

    # Calculate ratio j_n(z) / j_{n-1}(z) by downward recurrence for z = y
    j_ratio_y = np.zeros(y_ndown, dtype = np.cdouble)
    j_ratio_y_n = y / (2. * y_ndown + 1)

    n = y_ndown - 1
    while n > 0:
        ky_n = (2 * n + 1) * rcp_y
        j_ratio_y[n] = j_ratio_y_n = 1. / (ky_n - j_ratio_y_n)
        n -= 1

    # Variables for upward recurrences of Bessel fcts.
    jx_0 = np.sin(x) * rcp_x
    jy_0 = np.sin(y) * rcp_y

    # Variables for upward recurrences of Hankel fcts.
    h_exp = np.exp(1j * x) * rcp_x
    hx_0 = -1j * h_exp
    hx_1 = -h_exp * (1. + (1j * rcp_x))

    # Upward recurrence for deriv. of Legendre polynomial
    pi_0 = 0.
    pi_1 = 1.

    # Accumulation variables for S1 and S2 terms
    S1 = 0j
    S2 = 0j

    # Accumulation variables for Cs and Ct terms
    Cs = 0.
    Ct = 0.

    # Accumulation variable for normalization factor
    Ns = 0.

    n = 1
    while n <= x_nmax:
        j_ratio_x_n = j_ratio_x[n]
        j_ratio_y_n = j_ratio_y[n]

        # Upward recurrences for Bessel and Hankel functions
        if (n == 1):
            hx_n = hx_1
            hx_dx = x * hx_0 - n * hx_1
        else:
            hx_n = (2 * n - 1) * rcp_x * hx_1 - hx_0
            hx_dx = x * hx_1 - n * hx_n

            hx_0 = hx_1
            hx_1 = hx_n

        jx_n = j_ratio_x_n * jx_0
        jy_n = j_ratio_y_n * jy_0
        jx_dx = x * jx_0 - n * jx_n
        jy_dy = y * jy_0 - n * jy_n

        jx_0 = jx_n
        jy_0 = jy_n

        # Upward recurrences for angle-dependent terms based on
        # Legendre functions (Absorption and Scattering of Light by Small
        # Particles, Bohren and Huffman, p. 95)
        if (n == 1):
            pi_n = pi_1
            tau_n = mu
        else:
            pi_n = ((2 * n - 1) / (n - 1)) * mu * pi_1 - (n / (n - 1)) * pi_0
            tau_n = n * mu * pi_n - (n + 1) * pi_1
            
            pi_0 = pi_1
            pi_1 = pi_n

        # Lorenz-Mie coefficients (Eqs. 9, 10)
        a_n = (m_sq * jy_n * jx_dx - jx_n * jy_dy) / \
              (m_sq * jy_n * hx_dx - hx_n * jy_dy)
        b_n = (jy_n * jx_dx - jx_n * jy_dy) / \
              (jy_n * hx_dx - hx_n * jy_dy)

        a_norm = np.abs(a_n)
        b_norm = np.abs(b_n)
        a_sq_norm = a_norm * a_norm
        b_sq_norm = b_norm * b_norm

        # Calculate i-th term of S1 and S2
        kn = (2 * n + 1) / (n * (n + 1));
        S1 += kn * (a_n * tau_n + b_n * pi_n)
        S2 += kn * (a_n * pi_n + b_n * tau_n)

        # Calculate i-th term of Cs and Ct
        Cs += (2 * n + 1) * (a_sq_norm + b_sq_norm)
        Ct += np.real((2 * n + 1) * (a_n + b_n))

        # Calculate i-th term of factor in denominator
        Ns += (2 * n + 1) * (a_sq_norm + b_sq_norm)
    
        n += 1

    S1 *= 1j / kx
    S2 *= 1j / kx

    k = 2. * np.pi / kx_sq_norm
    Cs *= k
    Ct *= k

    Ns *= 0.5 / kx_sq_norm
    # Ns *= 0.5 * 4. * np.pi / kx_sq_norm

    return S1, S2, Ns, Cs, Ct

""" Calculates the Mueller matrix of a Mie scatter event given complex
scattering amplitudes of ordinary and extraordinary rays.

- s1
   Scattering amplitude/phase of the *ordinary* ray

- s2
   Scattering amplitude/phase of the *extraordinary* ray

- ns
   Normalization factor for phase function

Returns: 
   - A tuple containing all four non-zero Mueller matrix elements
"""
def mie_pf(s1, s2, ns):
    s1_norm = np.abs(s1)
    s2_norm = np.abs(s2)
    s1_sq_norm = s1_norm * s1_norm
    s2_sq_norm = s2_norm * s2_norm

    s11 = 0.5 * (s1_sq_norm + s2_sq_norm)
    s12 = 0.5 * (s1_sq_norm - s2_sq_norm)
    s33 = np.real(s1 * np.conj(s2))
    s34 = np.imag(s1 * np.conj(s2))

    return s11 / ns, s12 / ns, s33 / ns, s34 / ns