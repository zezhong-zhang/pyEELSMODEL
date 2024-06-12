import numpy as np
import pyEELSMODEL.api as em


# functions to make easy mask to make elemental maps
def make_circular_mask(xco, yco, Rin, Rout, shape):
    XX, YY = np.meshgrid(np.arange(shape[1]) - yco, np.arange(shape[0]) - xco)
    R = np.sqrt(XX ** 2 + YY ** 2)

    mask = np.zeros(shape)
    boolean = (R >= Rin) & (R < Rout)
    mask[boolean] = 1
    return mask


def make_rectangular_mask(xco, yco, width, height, shape):
    mask = np.zeros(shape)
    mask[xco:xco + width, yco:yco + height] = 1
    return mask


def simulate_data():
    """
    Small function which simulates a typical STEM-EELS map.

    """
    elements = ['C', 'N', 'O', 'Fe']
    edges = ['K', 'K', 'K', 'L']
    Zs = [6, 7, 8, 26]  # atomic weights

    # scan size elemental map
    xsize = 32
    ysize = 32
    maps = np.zeros((len(elements), xsize, ysize))

    # masks which define different regions.
    mask0 = make_rectangular_mask(5, 5, 20, 20, (xsize, ysize))
    mask1 = make_rectangular_mask(90, 90, 20, 30, (xsize, ysize))
    mask2 = make_circular_mask(xsize // 2, ysize // 2, 20, 30, (xsize, ysize))
    mask3 = make_circular_mask(xsize // 2, ysize // 2, 0, 20, (xsize, ysize))

    # attribute different elemental values for different masks
    maps[0] = 1  # carbon elemental map
    maps[1] = 2 * mask0 + mask1  # nitrogen elemental map
    maps[2] = mask2  # oxygen elemental map
    maps[3] = mask3 + 0.5 * mask2  # iron elemental map

    adf = np.zeros((xsize, ysize))
    tlambda_map = np.zeros_like(adf)
    for ii in range(maps.shape[0]):
        adf += (Zs[ii] ** 2 * maps[ii]) 
        tlambda_map += Zs[ii] * maps[ii]

    # maximum t/lambda is 1 for this sample
    tlambda_map = tlambda_map / tlambda_map.max()
    E0 = 300e3  # acceleration voltage [V]
    alpha = 1e-9  # convergence angle [rad]
    beta = 20e-3  # collection angle [rad]

    dispersion = 0.5  # [eV]
    offset = 200  # [eV]
    size = 2048  # number of pixel used in simulation [eV]

    settings = (E0, alpha, beta)
    msh = em.MultiSpectrumshape(dispersion, offset, size, xsize, ysize)

    sim = em.CoreLossSimulator(msh, elements, edges, maps, tlambda_map,
                               settings)
    # add shift which experimentally arises due to instabilities
    sim.use_shift = True
    sim.simulate_multispectrum()

    hl = sim.multispectrum
    ll = sim.ll
    # ll.multidata = ll.multidata 
    return hl, ll, maps

def plot_ground_truth(maps, labels =[ 'C', 'N', 'O', 'Fe']):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.ravel()
    cmap_colors = plt.cm.get_cmap('tab10')
    for ii in range(maps.shape[0]):
        base_color = cmap_colors(ii % cmap_colors.N)
        colors = [(1, 1, 1), base_color]
        cmap = LinearSegmentedColormap.from_list("custom_colormap", colors, N=256)
        im = axs[ii].imshow(maps[ii], cmap=cmap)
        axs[ii].set_title(f'{labels[ii]}')
        axs[ii].axis('off')
        plt.colorbar(im, ax=axs[ii], fraction=0.046, pad=0.04)
    plt.show()