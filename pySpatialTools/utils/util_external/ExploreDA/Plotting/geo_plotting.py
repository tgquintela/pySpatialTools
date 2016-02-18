
"""
Module which groups the geographical map plots.
"""


from mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import numpy as np

#from Mscthesis.Statistics.stats_functions import compute_spatial_density


def clean_coordinates(coordinates):
    # Delete nan
    coordinates = coordinates.dropna()
    coordinates = np.array(coordinates)
    # Delete 0,0
    idx = np.logical_and(coordinates[:, 0] != 0, coordinates[:, 1] != 0)
    coordinates = coordinates[idx, :]
    return coordinates


def plot_in_map(coordinates, resolution='f', color_cont=None, marker_size=1):
    """Plot the coordinates in points in the map.
    """

    # Delete nan
    coordinates = coordinates.dropna()
    coordinates = np.array(coordinates)
    # Delete 0,0
    idx = np.logical_and(coordinates[:, 0] != 0, coordinates[:, 1] != 0)
    coordinates = coordinates[idx, :]

    # compute coordinates
    longs = coordinates[:, 0]
    lats = coordinates[:, 1]

    lat_0 = np.mean(lats)
    llcrnrlon = np.floor(np.min(longs))
    llcrnrlat = np.floor(np.min(lats))
    urcrnrlon = np.ceil(np.max(longs))
    urcrnrlat = np.ceil(np.max(lats))

    # Set map
    fig = plt.figure()
    mapa = Basemap(projection='merc', lat_0=lat_0, llcrnrlon=llcrnrlon,
                   llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
                   urcrnrlat=urcrnrlat, resolution=resolution)
    mapa.drawcoastlines()
    mapa.drawcountries()
    if color_cont is not None:
        mapa.fillcontinents(color=color_cont)
    mapa.drawmapboundary()

    mapa.scatter(longs, lats, marker_size, marker='o', color='r', latlon=True)

    return fig


def plot_geo_heatmap(coordinates, n_levs, n_x, n_y, var=None):
    """Plot the coordinates in points in the map.
    """

    ## 00. Preprocess of the data
    # Delete nan
    coordinates = coordinates.dropna()
    coordinates = np.array(coordinates)
    # Delete 0,0
    idx = np.logical_and(coordinates[:, 0] != 0, coordinates[:, 1] != 0)
    coordinates = coordinates[idx, :]

    ## 0. Preparing needed variables
    # compute coordinates
    longs = coordinates[:, 0]
    lats = coordinates[:, 1]
    # Preparing corners
    lat_0 = np.mean(lats)
    llcrnrlon = np.floor(np.min(longs))
    llcrnrlat = np.floor(np.min(lats))
    urcrnrlon = np.ceil(np.max(longs))
    urcrnrlat = np.ceil(np.max(lats))

    ## 1. Set map
    fig = plt.figure()
    mapa = Basemap(projection='merc', lat_0=lat_0, llcrnrlon=llcrnrlon,
                   llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
                   urcrnrlat=urcrnrlat, resolution='h')
    mapa.drawcoastlines()
    mapa.drawcountries()
#    mapa.fillcontinents(color='gray')

    # Draw water
    #mapa.drawmapboundary(fill_color='aqua')
    #mapa.fillcontinents(color='coral')
    #mapa.drawlsmask(ocean_color='aqua', lakes=False)

    # mapa.scatter(longs, lats, 10, marker='o', color='k', latlon=True)

    ## 2. Preparing heat map
    density, l_x, l_y = compute_spatial_density(longs, lats, n_x+1, n_y+1, var)
    clevs = np.linspace(density.min(), density.max(), n_levs+1)
    l_x, l_y = mapa(l_x, l_y)

    ## 3. Computing heatmap
    #mapa.scatter(l_x, l_y, 1, marker='o', color='r', latlon=True)
    cs = mapa.contourf(l_x, l_y, density, clevs, cmap='Oranges')
    #cs = plt.contourf(l_x, l_y, density, clevs)
    # add colorbar.
    cbar = mapa.colorbar(cs, location='bottom', pad="5%")
    cbar.set_label('density')

    ## 4. Fix details

    # add title
    plt.title('Heat map of companies density')

    return fig


###############################################################################
###############################################################################
###############################################################################
def compute_spatial_density(longs, lats, n_x, n_y, var=None, sigma_smooth=5,
                            order_smooth=0):
    """Computation of the spatial density given the latitutes and logitudes of
    the points we want to count.

    TODO
    ----
    Smoothing function

    """
    ## 0. Setting initial variables
    llcrnrlon = np.floor(np.min(longs))
    llcrnrlat = np.floor(np.min(lats))
    urcrnrlon = np.ceil(np.max(longs))
    urcrnrlat = np.ceil(np.max(lats))
    l_x = np.linspace(llcrnrlon, urcrnrlon, n_x+1)
    l_y = np.linspace(llcrnrlat, urcrnrlat, n_y+1)

    ## 1. Computing density
    density = computing_density_var(longs, lats, [l_x, l_y], var)
    density = density.T

    ## 2. Smothing density
    density = ndimage.gaussian_filter(density, sigma_smooth, order_smooth)

    ## 3. Output
    l_x = np.mean(np.vstack([l_x[:-1], l_x[1:]]), axis=0)
    l_y = np.mean(np.vstack([l_y[:-1], l_y[1:]]), axis=0)
    l_x, l_y = np.meshgrid(l_x, l_y)

    return density, l_x, l_y


def compute_spatial_density_sparse(longs, lats, n_x, n_y, null_lim=0.1,
                                   var=None):
    """Computation of the spatial density given the latitutes and logitudes of
    the points we want to count.

    TODO
    ----
    Smoothing function

    """
    ## 0. Setting initial variables
    llcrnrlon = np.floor(np.min(longs))
    llcrnrlat = np.floor(np.min(lats))
    urcrnrlon = np.ceil(np.max(longs))
    urcrnrlat = np.ceil(np.max(lats))
    l_x = np.linspace(llcrnrlon, urcrnrlon, n_x+1)
    l_y = np.linspace(llcrnrlat, urcrnrlat, n_y+1)

    ## 1. Computing density
    density = computing_density_var(longs, lats, [l_x, l_y], var)
    #density = density.T

    ## 2. Smothing density

    ## 3. Output
    l_x = np.mean(np.vstack([l_x[:-1], l_x[1:]]), axis=0)
    l_y = np.mean(np.vstack([l_y[:-1], l_y[1:]]), axis=0)

    idxs = (density > null_lim).nonzero()
    density = density[idxs]

    l_x = l_x[idxs[0]]
    l_y = l_y[idxs[1]]

    return density, l_x, l_y


def computing_density_var(longs, lats, border_grid, var=None):
    """"""
    if var is None:
        density, _, _ = np.histogram2d(longs, lats, border_grid)
    else:
        l_x_bor = border_grid[0]
        l_y_bor = border_grid[1]
        n_x = l_x_bor.shape[0]-1
        n_y = l_y_bor.shape[0]-1

        ## Indexing part
        density = np.zeros((n_x, n_y))
        for i in range(n_x):
            idxs_i = np.logical_and(l_x_bor[i] <= longs, l_x_bor[i+1] >= longs)
            for j in range(n_y):
                idxs_j = np.logical_and(l_y_bor[j] <= lats,
                                        l_y_bor[j+1] >= lats)
                idxs = np.logical_and(idxs_i, idxs_j)
                density[i, j] = np.sum(var[idxs])
    return density
