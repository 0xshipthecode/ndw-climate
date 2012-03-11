

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np


def render_component(m, C, lats, lons, rmax, sym_clims = False, title = None):
    """
    Render a single component onto the current axes.  Lats/lons must
    be sorted and must correspond to the 2D component C.  The image
    will be drawn into the given axes. A colorbar is drawn automatically,
    a title is set optionally.
    """
    m.drawcoastlines()
    m.etopo(scale = 0.2)
    m.drawparallels(np.arange(-90.,91.,30.))
    m.drawmeridians(np.arange(0.,361.,60.))
    
    nx = int((m.xmax-m.xmin) / 20000) + 1
    ny = int((m.ymax-m.ymin) / 20000) + 1
    f = m.transform_scalar(C, lons, lats, nx, ny)
    
    imgplt = m.imshow(f, alpha = 0.8)
    if sym_clims:
        imgplt.set_clim(-rmax, rmax)
        
    plt.colorbar()
    
    if title != None:
        plt.title(title)
        
    return imgplt
        
        
def render_component_single(C, lats, lons, sym_clims = False, fname = None, plt_name = None):
    """
    Render a single component on a plot.
    """
    m = Basemap(projection='mill',
                llcrnrlat=min(lats), urcrnrlat=max(lats),
                llcrnrlon=(min(lons)), urcrnrlon=max(lons),
                resolution='c')
    
    rmax = max([np.max(C), np.max(-C)])
    
    if plt_name == None:
        plt_name = 'Component'

    # in case lats are not in ascending order, fix this
    lat_ndx = np.argsort(lats)
    lats_s = lats[lat_ndx]
        
    plt.figure(figsize = (20,8))
    plt.subplot(1, 1, 1)
    render_component(m, C[lat_ndx, :], lats_s, lons, rmax, sym_clims, plt_name)
    
    if fname:
        plt.savefig(fname)
    

def render_component_triple(Cd, Cmn, Cmx, lats, lons, fname = None, plt_name = None):
    """
    Render a component triple, the component, its max and min deviation.  Everything is rendered
    with the same color range.
    """
    m = Basemap(projection='mill',
                llcrnrlat=min(lats), urcrnrlat=max(lats),
                llcrnrlon=(min(lons)), urcrnrlon=max(lons),
                resolution='c')
    
    rmax = max([np.max(Cmx), np.max(-Cmx)])
    
    if plt_name == None:
        plt_name = 'Component'

    # in case lats are not in ascending order, fix this
    lat_ndx = np.argsort(lats)
    lats_s = lats[lat_ndx]
        
    plt.figure(figsize = (20,20))
    plt.subplot(3, 1, 1)
    render_component(m, Cmn[lat_ndx, :], lats_s, lons, rmax, plt_name + ' - min')
    plt.subplot(3, 1, 2)
    render_component(m, Cd[lat_ndx, :], lats_s, lons, rmax, plt_name + ' - data')
    plt.subplot(3, 1, 3)
    render_component(m, Cmx[lat_ndx, :], lats_s, lons, rmax, plt_name + ' - max')
    
    if fname:
        plt.savefig(fname)


def render_components(C, lats, lons, fname_tmpl = None, ndx = None):
    """
    Render the components in C [with dims comp_id x lats x lons] onto
    a world map of appropriate size.  The method assumes that the  lons x lats
    generate a rectangle.
    """
    m = Basemap(projection='mill',llcrnrlat=min(lats), urcrnrlat=max(lats), llcrnrlon=(min(lons)),
                urcrnrlon=max(lons),resolution='c')
    
    if ndx == None:
        ndx = np.arange(len(C)) + 1
        
    rmax = np.max(C)

    # lattitudes may not be sorted in ascending order, rectify this
    lat_ndx = np.argsort(lats)
    lats_s = lats[lat_ndx]

    # render each topo plot
    for ci in range(len(C)):
        
        plt.figure(figsize=(12, 8 * (max(lats) - min(lats)) / 180))
        plt.axes([0.05, 0.05, 0.9, 0.85])

        m.drawcoastlines()
        #m.fillcontinents(color='coral',lake_color='aqua', zorder = 0)
        m.etopo(scale = 0.2)

        # draw parallels and meridians.
        m.drawparallels(np.arange(-90.,91.,30.))
        m.drawmeridians(np.arange(0.,361.,60.))
        #m.drawmapboundary(fill_color='aqua')
        
        nx = int((m.xmax-m.xmin) / 20000) + 1
        ny = int((m.ymax-m.ymin) / 20000) + 1
        Ci = C[ci, lat_ndx, :]
        f = m.transform_scalar(Ci, lons, lats_s, nx, ny)
        
        # imlim seems to be equivalent to caxis() im MATLAB
        imgplt = m.imshow(f, alpha = 0.8)
        imgplt.set_clim(-rmax, rmax)
        plt.colorbar()
        
        plt.title('Component %d' % (ndx[ci]))
        
        if fname_tmpl:
            plt.savefig(fname_tmpl % (ndx[ci]))
            
    if not fname_tmpl:
        plt.show()
    
