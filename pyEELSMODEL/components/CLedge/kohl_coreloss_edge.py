from pyEELSMODEL.misc import hs_gdos as hsdos
from pyEELSMODEL.components.CLedge.coreloss_edge import CoreLossEdge
from pyEELSMODEL.misc.elements_list import elements
import os
import h5py

class KohlLossEdge(CoreLossEdge):
    """
    Coreloss edges which are calculated by Leonhard Segger, Giulio Guzzinati
    and Helmut Kohl https://zenodo.org/record/6599071#.Y3I1cnbMKUk



    """
    def __init__(self, specshape, A, E0, alpha, beta, element, edge, eshift=0, q_steps=100):
        # dir path should be defined before set_edge function is called
        # use relative paths
        # self.dir_path = r'..\H-S GOS Tables'
        self.set_dir_path(r'C:\Users\DJannis\Documents\KohlCrossSection\Segger_Guzzinati_Kohl_1.5.0.gosh')
        self.set_onset_path(r'C:\Users\DJannis\Documents\KohlCrossSection\onset_energies.hdf5')
        super().__init__(specshape, A, E0, alpha, beta, element, edge, eshift=eshift, q_steps=q_steps)

        self.set_gos_energy_q()
        #redefine some attributes


    def set_gos_energy_q(self):
        if int(self.edge[-1]) == 1:
            edge = self.edge
        elif (int(self.edge[-1]) % 2 == 0):
            edge = self.edge + str(int(self.edge[-1]) + 1)
        else:
            edge = self.edge[0] + str(int(self.edge[-1]) - 1) + self.edge[-1]

        with h5py.File(self.dir_path, 'r+') as f:
            self.gos = f[self.element][edge]['data'][:]
            self.free_energies = f[self.element][edge]['free_energies'][:]
            self.q_axis = f[self.element][edge]['q'][:]

    def set_element(self, element):
        with h5py.File(self.dir_path, 'r+') as f:
            elem_list = list(f.keys())
        if element in elem_list:
            self.element = element
        else:
            raise ValueError('Element you selected is not valid')

    def set_dir_path(self, path):
        self.dir_path = path

    def set_onset_path(self, path):
        self.onset_path = path

    # def set_onset_energy(self):
    #     with h5py.File(self.onset_path, 'r+') as f:
    #         data = f[self.element][self.edge][:]
    #         self.onset_energy = data[0] + self.eshift
    #         self.prefactor = data[1]

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
        edge_list = ['K1', 'L1', 'L2', 'L3', 'M1','M2', 'M3', 'M4', 'M5',
                     'N1','N2', 'N3', 'N4', 'N5', 'N6', 'N7']
        if not isinstance(edge, str):
            raise TypeError('Edge should be a string: K1, L1, L2, L3, M2, M3,'
                            ' M4, M5, N4, N5', 'N6', 'N7')
        if edge in edge_list:
            self.edge = edge
        else:
            raise ValueError('Edge should be: K1, L1, L2, L3, M2, M3, M4, M5,'
                             ' N4, N5', 'N6', 'N7')

    def set_edge(self, edge):
        """
        Checks if the given edge is valid and adds the directories of the
        :param edge:
        :return:
        """
        edge_list = ['K1', 'L1', 'L2', 'L3', 'M1','M2', 'M3', 'M4', 'M5',
                     'N1','N2', 'N3', 'N4', 'N5', 'N6', 'N7']

        elem_list = self.get_elements()

        if not isinstance(edge, str):
            raise TypeError('Edge should be a string: K1, L1, L2, L3, M2, M3,'
                            ' M4, M5, N4, N5', 'N6', 'N7')
        if edge in edge_list:
                self.edge = edge
        else:
            raise ValueError('Edge should be: K1, L1, L2, L3, M2, M3, M4, M5,'
                             ' N4, N5', 'N6', 'N7')

    def calculate_cross_section(self):
        """
        Calculates the cross section in barns


        """
        ek = self.onset_energy
        E0 = self.parameters[1].getvalue()
        alpha = self.parameters[3].getvalue()
        beta = self.parameters[2].getvalue()
        e_axis = self.free_energies
        q_axis = self.q_axis
        gos = self.gos

        prf = 1e28 * self.prefactor #prefactor and convert to barns
        css = prf  *  hsdos.dsigma_dE_from_GOSarray(self.energy_axis,
                                                    e_axis+ek, ek, E0, beta,
                                                    alpha, q_axis, gos,
                                                    q_steps=100,
                                                    swap_axes=True)

        return css












