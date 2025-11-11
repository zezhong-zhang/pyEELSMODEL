from pyEELSMODEL.misc import hs_gdos as hsdos
from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge
from pyEELSMODEL.database.Zhang.download import download_file
import os
import h5py
from pyEELSMODEL import __file__
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

class ZezhongCoreLossEdge(CoreLossEdge):
    """
    Coreloss edges which are calculated by Zezhong Zhang.
    https://zenodo.org/records/11199911

    """

    def __init__(self, specshape, A, E0, alpha, beta, element, edge, eshift=0,
                 q_steps=100, dir_path=None, include_white_lines=False, energy_resolution=0.5):
        """

        Parameters
        ----------
        specshape : Spectrumshape
            The spectrum shape used to model
        A : float
              Amplitude of the edge

        E0: float [V]
              The acceleration voltage of the microscope

        alpha: float [rad]
              The convergence angle of the incoming probe

        beta: float [rad]
              The collection angle

        element: string
              The name of the element from which to calculate the edge model.

        edge: string
              The type of edge. (K, L or M)

        eshift: float [eV]
              The shift of the onset energy with respect to the literature
              value.
            (default: 0)

        q_steps: int
             The number of q points taken into account for the integration
             over the momentum space. The larger the number of q_steps the more
             accurate the calculation. (default: 100)

        include_white_lines: bool
             Whether to include white lines (discrete bound states) in the calculation.
             (default: False)

        energy_resolution: float [eV]
             Energy resolution for Gaussian broadening of white lines.
             (default: 0.5)

        Returns
        -------
        """
        if dir_path is None:
            self.dir_path = os.path.dirname(
                os.path.dirname(__file__) + "/../pyEELSMODEL/database/Zhang/"
            )
            self.file = os.path.join(self.dir_path, 'Dirac_GOS.gosh')    
        else:
            self.set_dir_path(dir_path)
        if not os.path.exists(self.file):
                download_file(filename=self.file)        
        if edge == 'K':
            edge = 'K1'

        self.include_white_lines = include_white_lines
        self.energy_resolution = energy_resolution
        self.white_lines_file = os.path.join(self.dir_path, 'Dirac_GOS_white_lines.gosh')

        super().__init__(specshape, A, E0, alpha, beta, element, edge,
                         eshift=eshift, q_steps=q_steps)
        self.set_gos_energy_q()
        self.swap_axes = False
        # an approximation for the GOS which does not do the full integration
        self.use_approx = False
        
        # Load white lines data if requested
        if self.include_white_lines:
            self._load_white_lines_data()

    def set_gos_energy_q(self):
        with h5py.File(self.file, 'r') as f:
            self.gos = f[self.element][self.edge]['data'][:].squeeze().T
            self.free_energies = f[self.element][self.edge]['free_energies'][:][
                                 :]  # two dimensional array
            self.q_axis = f[self.element][self.edge]['q'][
                          :] # in  [1/m]
            self.ionization_energy = f[self.element][self.edge]['metadata'].attrs['ionization_energy']

    def set_element(self, element):
        self.element = element

        # file = os.path.join(self.dir_path, element + '.hdf5')
        # isExist = os.path.exists(file)
        # if not isExist:
        #     raise ValueError('Element you selected is not valid')
        # self.element = element
        # self.dir_path_element = file  # the filename to have easy access

    def set_dir_path(self, path):
        self.dir_path = path
        self.file = os.path.join(path, 'Dirac_GOS.gosh')

    def _check_str_in_list(self, list, edge):
        for name in list:
            if name[0] == edge[0]:
                return True

        return False

    def set_edge(self, edge):
        """
        Checks if the given edge is valid and adds the directories of the
        :param edge:
        :return:
        """
        edge_list = ['K1', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5',
                     'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
        if not isinstance(edge, str):
            raise TypeError(
                'Edge should be a string: K1, L1, L2, L3, M2, M3, M4, M5, N4, '
                'N5,N6, N7')
        if edge in edge_list:
            self.edge = edge
        else:
            raise ValueError(
                'Edge should be: K1, L1, L2, L3, M2, M3, M4, M5, N4, N5',
                'N6, N7')

    def _load_white_lines_data(self):
        """Load white lines data from HDF5 file"""
        if not os.path.exists(self.white_lines_file):
            print(f"White lines file not found: {self.white_lines_file}")
            self.include_white_lines = False
            return
        
        try:
            with h5py.File(self.white_lines_file, 'r') as f:
                edge_group = f[self.element][self.edge]
                
                if "white_lines" not in edge_group.keys():
                    print(f"No white lines data for {self.element} {self.edge}")
                    self.include_white_lines = False
                    return
                
                wl_group = edge_group["white_lines"]
                self.gos_wl = wl_group["data"][:].squeeze()
                self.free_energies_wl = wl_group["free_energies"][:].squeeze()
                self.q_wl = wl_group["q"][:].squeeze() * 1e10  # Convert 1/A to 1/m
        except Exception as e:
            print(f"Error loading white lines: {e}")
            self.include_white_lines = False

    def _calculate_white_lines_cross_section(self, E0, alpha, beta):
        """Calculate cross section for discrete bound states (white lines)"""
        R = hsdos.pc.R()
        T = hsdos.pc.T(E0)
        gamma = hsdos.pc.gamma(E0)
        a0 = hsdos.pc.a0()
        prf = self.prefactor * 1e28
        
        discrete_energies = []
        discrete_cross_sections = []
        
        for i, epsilon in enumerate(self.free_energies_wl):
            E_loss = self.onset_energy + epsilon
            discrete_energies.append(E_loss)
            
            GOSarray = self.gos_wl[i, :]
            qa0sq_min, qa0sq_max = hsdos.hdos.get_qmin_max(E_loss, E0, beta, alpha=alpha)
            
            def integrand(log_qa0sq):
                qa0sq = np.exp(log_qa0sq)
                q = np.sqrt(qa0sq) / a0
                theta = 2. * np.sqrt(np.abs(R * (qa0sq - qa0sq_min) / (4. * gamma ** 2 * T)))
                
                if q < self.q_wl.min() or q > self.q_wl.max():
                    df_dE = 0
                else:
                    df_dE = np.interp(q, self.q_wl, GOSarray)
                
                return df_dE * hsdos.hdos.convergence_correction_factor(alpha, beta, theta)
            
            integral, _ = quad(integrand, np.log(qa0sq_min), np.log(qa0sq_max), limit=100)
            sigma = 4 * np.pi * a0**2 * (R / E_loss) * (R / T) * integral * prf
            discrete_cross_sections.append(sigma)
        
        # Create spectrum with Gaussian broadening
        discrete_spectrum = np.zeros_like(self.energy_axis)
        for E_discrete, sigma_discrete in zip(discrete_energies, discrete_cross_sections):
            gaussian = np.exp(-0.5 * ((self.energy_axis - E_discrete) / self.energy_resolution) ** 2)
            gaussian /= (self.energy_resolution * np.sqrt(2 * np.pi))
            discrete_spectrum += gaussian * sigma_discrete 
        
        return discrete_spectrum, np.array(discrete_energies)

    def calculate_cross_section(self):
        ek = self.onset_energy
        E0 = self.parameters[1].getvalue()
        alpha = self.parameters[3].getvalue()
        beta = self.parameters[2].getvalue()
        e_axis = self.free_energies
        q_axis = self.q_axis
        gos = self.gos
        pref = 1e28 * self.prefactor

        # Calculate continuum cross section
        if self.use_approx:
            cross_section = pref * hsdos.dsigma_dE_from_GOSarray_approx(
                self.energy_axis, e_axis + ek, E0, beta, gos)
        else:
            cs = hsdos.dsigma_dE_from_GOSarray_bound(self.energy_axis, e_axis,
                                                     ek, E0, beta, alpha,
                                                     q_axis, gos, q_steps=100)
            cross_section = pref * cs

        # Add white lines if requested
        if self.include_white_lines:
            discrete_spectrum, discrete_energies = self._calculate_white_lines_cross_section(E0, alpha, beta)
            # Only add continuum above the last bound state
            mask_continuum = self.energy_axis > discrete_energies.max()
            combined = discrete_spectrum.copy()
            combined[mask_continuum] += cross_section[mask_continuum]
            cross_section = combined

        return cross_section
    
    @property
    def E0(self):
        return self.parameters[1].getvalue()

    @property
    def k0(self):
        return 2*np.pi/hsdos.energy2wavelength(self.E0) 

    def plot_gos(self):
        plt.figure()
        plt.pcolormesh( self.free_energies, self.gos)
        plt.xlabel('q [1/A]')
        plt.ylabel('Energy above ionization [eV]')
        plt.colorbar()
        plt.title('GOS for ' + self.element + ' ' + self.edge)
        plt.show()

    def plot_ddscs(self):
        ddscs = hsdos.ddscs_dE_dOmega(self.free_energies, self.onset_energy,self.E0,self.q_axis, self.gos)
        plt.subplots(1, 2)
        plt.subplot(121)
        plt.pcolormesh(self.q_axis/1e10, self.free_energies, ddscs)
        plt.ylabel('Energy loss [eV]')
        plt.xlabel('Scattering vector [1/A]')
        plt.xlim([0, 1])
        plt.colorbar()
        plt.tight_layout()
        plt.subplot(122)
        theta = np.arctan(self.q_axis/1e10 /self.k0)*1e3 # this is approximation
        plt.pcolormesh(theta, self.free_energies, ddscs) 
        plt.xlim([0, 10])
        plt.xlabel('Scattering angle [mrad]')
        plt.ylabel('Energy loss [eV]')
        plt.colorbar()  
        plt.tight_layout()