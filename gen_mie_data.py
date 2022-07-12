import os
import numpy as np
from tqdm import tqdm

from mie import mie, mie_pf

file_dir = 'mie_data_mono_script/'

def format_float(f):
    f = f.replace('_', '.')
    f = f.replace('D', 'e')
    return float(f)

def eval_mie_all():
    print_files = [ f for f in os.listdir(file_dir) if f.endswith('.npy') ]
    print_files = [ f for f in print_files if f.find('_mts') == -1 ]

    for f in tqdm(print_files):
        ext_idx  = f.find('.npy')
        wav_idx  = f.find('w')
        rad_idx  = f.find('r')
        imag_idx = f.find('i')

        wav_str  = f[wav_idx+1 : rad_idx]
        rad_str  = f[rad_idx+1 : imag_idx]
        imag_str = f[imag_idx+1 : ext_idx]

        # Get matching parameters to Mishchenko files
        wavelength = format_float(wav_str) * 1000.0
        radius = format_float(rad_str) * 1000.0
        ior_sph_i = format_float(imag_str)

        ior_med = 1. + 0j
        ior_sph = 1.53 + ior_sph_i
        nmax = -1

        # Evaluate phase function for range of angles
        n = 180
        thetas = np.linspace(0.0, np.pi, n)
        mus = np.cos(thetas)

        s11s = np.zeros(n)
        s12s = np.zeros(n)
        s33s = np.zeros(n)
        s34s = np.zeros(n)

        for i, mu in enumerate(mus):
            s1, s2, ns, _, _ = mie(radius, wavelength, ior_sph, ior_med, mu, nmax)
            s11, s12, s33, s34 = mie_pf(s1, s2, ns)

            s11s[i] = s11
            s12s[i] = s12
            s33s[i] = s33
            s34s[i] = s34

        # When not visually comparing against Mishchenko results,
        # also need to normalize (divide) phase function values by 4pi

        # Normalize by s11 (only for visualization)
        s12s /= s11s 
        s33s /= s11s
        s34s /= s11s

        theta_deg = np.rad2deg(thetas)
        data = np.column_stack((theta_deg, s11s, s12s, s33s, s34s))
        # print(data_mts)

        np.save(file_dir + f[:f.rfind('.')] + '_ours.npy', data)

if __name__ == '__main__':
    eval_mie_all()