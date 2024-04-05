import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable
import numpy as np

from scipy.stats import gaussian_kde
from other_functions import get_cell_coordinates

from datetime import datetime

def plot_earthquakes_time_histogram(database):
    plt.hist(database['date'], bins='auto', edgecolor='black')

    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Earthquakes grouped by Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_earthquakes_mag_histogram(earthquake_db):

    plt.hist(earthquake_db['mag'], bins='auto', edgecolor='black')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.title('Histogram of Magnitude')
    plt.xlim(2,max(earthquake_db['mag'])+.5)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_earthquakes_mag_histogram_bins(earthquake_db, bin_width = 0.25):

    # Calculate the number of bins dynamically based on the data range and bin width
    min_mag = earthquake_db['mag'].min()
    max_mag = earthquake_db['mag'].max()
    num_bins = int((max_mag - min_mag) / bin_width) + 1

    # Plot histogram with the specified bin width
    plt.hist(earthquake_db['mag'], bins=num_bins, range=(min_mag, max_mag), edgecolor='black')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Magnitude (Bin Width = {bin_width})')
    plt.xlim(min(earthquake_db['mag'])-.5,max(earthquake_db['mag'])+.5)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_earthquakes_depth_histogram(earthquake_db):
    plt.hist(earthquake_db['depth'], bins='auto', edgecolor='black')

    plt.xlabel('Depth')
    plt.ylabel('Frequency')
    plt.title('Histogram of Depth')
    plt.xlim(earthquake_db['depth'].min()-.5,earthquake_db['depth'].max()+.5)
    plt.grid(True)
    # plt.yscale('log')
    plt.tight_layout()
    plt.show()

def plot_earthquakes_depth_histogram_bins(earthquake_db, n_bins):

    bin_edges = np.percentile(earthquake_db['depth'], np.linspace(0, 100, n_bins + 1))
    plt.hist(earthquake_db['depth'], bins=bin_edges, edgecolor='black')

    plt.xlabel('Depth')
    plt.ylabel('Frequency')
    plt.title('Histogram of Depth with Equal-Frequency Binning')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def load_subdivisions_data(earthquake_db):
    subdivisions = {
        'Sedimentary Crust': [(-10,35,"blue"),(-10, 5, "lightgray"), (5, 15, "gray"), (15, 35, "darkgray")],
        'Transitional Crust': [(35,70,"green"),(35, 45, "lightgray"), (45, 60, "gray"), (60, 70, "darkgray")],
        'Lower Crust': [(70,250,"yellow"),(70, 100, "lightgray"), (100, 150, "gray"), (150, 250, "darkgray")],
        'Mantle Transition Zone': [(250,410,"orange"),(250, 330, "lightgray"), (330, 410, "gray")],
        'Upper Mantle': [(410,earthquake_db['depth'].max(),"red"),(410, 500, "lightgray"), (500, 600, "gray"), (600, earthquake_db['depth'].max(), "darkgray")]
    }
    return subdivisions

def plot_earthquakes_depth_histogram_log(earthquake_db,biggest_earthquakes,treshold_year,bin_width = 3.5):
    def log_format(x, pos):
        return f'$10^{{{x:.0f}}}$'

    subdivisions = load_subdivisions_data(earthquake_db)

    plt.figure(figsize=(15, 6))

    for layer_name, layer_data in subdivisions.items():
        global_layer_data = layer_data[0]
        min_depth,max_depth,color = global_layer_data
        plt.axvspan(min_depth, max_depth, alpha=0.2, color=color, label=layer_name)

        for sub_layer_data in layer_data[1:]:
            min_depth, max_depth, color = sub_layer_data
            plt.axvspan(min_depth, max_depth, alpha=0.2, color=color)#, label=layer_name)


    depth_data = earthquake_db['depth']

    num_bins = int((depth_data.max() - depth_data.min()) / bin_width)

    hist, _, patches = plt.hist(depth_data, bins=num_bins, edgecolor='black', range=(depth_data.min(), depth_data.max()))

    cmap = plt.get_cmap('RdYlGn')

    log_hist = np.log10(hist + 1e-3) #to avoid log(0)
    norm = plt.Normalize(log_hist.min(), log_hist.max())

    for log_count, patch in zip(log_hist, patches):
        color = cmap(norm(log_count))
        patch.set_facecolor(color)

    for i, eq in biggest_earthquakes.iterrows():
        depth = eq["depth"]
        if i ==0:
            plt.vlines(x=depth,ymin=0,ymax=log_hist.max()*10, color = 'red', linestyles= 'dashed', label = f"{len(biggest_earthquakes)} Major Earthquakes")
        else:
            plt.vlines(x=depth,ymin=0,ymax=log_hist.max()*10, color = 'red', linestyles= 'dashed')

    plt.xlabel('Depth')
    plt.ylabel('Frequency')
    plt.title(f'Earthquakes of magnitude >= 2.5 depth since {treshold_year} (log scale)')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, label='Frequency (Log Scale)')
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(log_format))
    cbar.set_label('Frequency (Log Scale)')

    plt.grid(False)
    plt.xlim(-bin_width, earthquake_db["depth"].max() + bin_width)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()

def custom_marker_size(magnitude,max_mag):
    return (100 * (np.log10(1 + magnitude) / np.log10(1 + max_mag))) ** 1.3


def plot_earthquakes(filtered_earthquakes, min_mag, colormap, norm, world, plot_world, title):

    if plot_world:
        _, ax = plt.subplots(figsize=(15, 12))
        world.boundary.plot(ax=ax, color='black')
    else:
        _, ax = plt.subplots(figsize=(12, 6))

    sc = ax.scatter(
        filtered_earthquakes['longitude'],
        filtered_earthquakes['latitude'],
        alpha=0.4,
        s=filtered_earthquakes['marker_size'],
        c=filtered_earthquakes['mag'],
        cmap=colormap,
        label=f"Magnitude > {min_mag}",
        norm=norm
    )
    # We also plot the location of Edinburgh to see if the data is accurate
    ax.scatter(-3.19648,55.95206,c='r', marker="x", label=f" Edinburgh ")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    cbar = plt.colorbar(sc, ax=ax, label='Magnitude', shrink=0.6)

    plt.tight_layout()

    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()

def plot_earthquakes_on_lat_long(filtered_earthquakes, min_mag, colormap, norm, world, plot_world):
    title = f'Earthquakes with Magnitude > {min_mag}'
    plot_earthquakes(filtered_earthquakes, min_mag, colormap, norm, world, plot_world, title)


def plot_earthquakes_on_worldmap(filtered_earthquakes, min_mag, colormap, norm,  world, plot_world):
    title = f'Earthquakes with Magnitude > {min_mag} on World Map'
    plot_earthquakes(filtered_earthquakes, min_mag, colormap, norm, world, plot_world, title)


def assign_dim(mag_power):
    if mag_power == 1:
        return ''
    elif mag_power ==2 :
        return 'quadratic '
    elif mag_power == 3:
        return 'cubic '
    elif mag_power > 3:
        return 'high power '

def dispatch_shocks(cluster):
    significant_shocks = cluster[cluster['mag']>=6.]
    medium_shocks = cluster[(cluster['mag']<6.) & (cluster['mag']>=5.)]
    unsignificant_shocks = cluster[cluster['mag']<5.]

    return unsignificant_shocks, medium_shocks, significant_shocks

def density_func(x, y, cluster, mag_power):
    xx, yy = np.meshgrid(x, y)
    kde = gaussian_kde(np.vstack([cluster['longitude'], cluster['latitude']]), weights= cluster['mag']**mag_power)

    # Calculate the density for each point in the grid
    density = kde(np.vstack([xx.ravel(), yy.ravel()]))
    density = density / density.max()  # Normalize the density values
    return density.reshape(xx.shape)

def shock_text(cluster, aftershocks,max_magnitude_index, mag_med):
    if len(cluster.iloc[:max_magnitude_index])==1:
        foreshock_text = '1 foreshock'
    elif len(cluster.iloc[:max_magnitude_index])>1:
        foreshock_text = f'{len(cluster.iloc[:max_magnitude_index])} foreshocks'
    else:
        foreshock_text = 'No foreshock'

    if len(aftershocks)==1:
        aftershock_text = f'1 aftershock of mag {mag_med}'
    elif len(aftershocks)>1:
        aftershock_text = f'{len(aftershocks)} aftershocks of median mag {mag_med}'
    else:
        aftershock_text = f'no aftershock'

    return foreshock_text, aftershock_text

def plot_cluster(world, grid, cluster, main_earthquake, aftershocks, max_magnitude_index, mag_med, near_area, degree = 8,mag_power = 2):
    #TO ADD: retrieve (min,max) for longitude and latitude, plot +/-1° compared to this values
    foredegree, afterdegree = degree,degree
    resolution = 5e-3

    date_first_shock = cluster.iloc[0].time
    date_first_shock = datetime.fromisoformat(date_first_shock.split('T')[0]).strftime('%Y-%m-%d')
    date_ms   = main_earthquake.time
    date_ms = datetime.fromisoformat(date_ms.split('T')[0]).strftime('%Y-%m-%d')
    date_last_as = aftershocks.iloc[-1].time
    date_last_as = datetime.fromisoformat(date_last_as.split('T')[0]).strftime('%Y-%m-%d')
    max_delta_day = aftershocks.iloc[-1].delta_days

    unsignificant_shocks, medium_shocks, significant_shocks = dispatch_shocks(aftershocks)

    if len(significant_shocks) >10:
        max_sig_shock_distance = significant_shocks.delta_dist.max()
    else:
        max_sig_shock_distance = aftershocks.delta_dist.max()

    dim = assign_dim(mag_power)

    mag_main = main_earthquake.mag

    main_earthquake_grid_cell = main_earthquake.grid_cell
    if isinstance(main_earthquake_grid_cell, str):
        main_earthquake_grid_cell = tuple(map(int, main_earthquake_grid_cell.strip('()').split(',')))
    i,j = main_earthquake_grid_cell[0],main_earthquake_grid_cell[1]

    min_lat,min_lon,max_lat,max_lon = get_cell_coordinates(grid,i,j)

    unzoom_data = int(max_sig_shock_distance/111)+1 # 1°of latitude or longitude is approximately equals to 111 km then we transform the distance in km to °
    x = np.arange(min_lon-unzoom_data, max_lon+unzoom_data, resolution)
    y = np.arange(min_lat-unzoom_data, max_lat+unzoom_data, resolution)

    density = density_func(x, y, cluster, mag_power)

    cmap_density = 'jet'
    norm_density = plt.Normalize(vmin=density.min(), vmax=density.max())
    smd = plt.cm.ScalarMappable(cmap=cmap_density, norm=norm_density)
    smd.set_array([])  # This line is necessary to connect the colormap to the data


    _, (ax1, ax2) = plt.subplots(1,2,figsize=(16, 8))

    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(y.min(), y.max())

    world.boundary.plot(ax=ax1, color='black', alpha = 0.75)
    world.plot(ax=ax1, color="brown", alpha=0.2)

    ax1.imshow(density, cmap=cmap_density, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', alpha=0.85,  aspect="auto")

    ax1.plot([], [], color="brown", alpha=0.2, label="Land") #to add the land label

    ax1.plot(unsignificant_shocks.longitude, unsignificant_shocks.latitude, marker = ".", markersize= 2, color = 'grey', alpha = 0.5 , linestyle='')
    ax1.plot(medium_shocks.longitude, medium_shocks.latitude, marker = "o", markersize=3, color = 'lightgray', alpha = 0.8, linestyle='')

    signshock_text = f"{len(significant_shocks)} significant shocks, mag ≥ 6" if len(significant_shocks) > 1 else f"{len(significant_shocks)} significant shock, mag ≥ 6"

    ax1.plot(significant_shocks.longitude, significant_shocks.latitude, marker = "^", markersize=7, color = 'white', linestyle='', label = f"{signshock_text}")
    ax1.plot(main_earthquake.longitude, main_earthquake.latitude, marker = '+', color = 'black', label = f'Main Shock\nof Mag = {mag_main}')

    ax1.hlines(y=main_earthquake['latitude'], xmin= x.min(), xmax= x.max(), color = 'black', linestyles= 'dashed')
    ax1.vlines(x=main_earthquake['longitude'], ymin= y.min(), ymax= y.max(), color = 'black', linestyles= 'dashed')

    ax1.set_xlabel('Longitude')  # Use set_xlabel for x-axis label
    ax1.set_ylabel('Latitude')   # Use set_ylabel for y-axis label

    ax1.legend()
    ax1.grid(False)
    ax1.set_title(f'Cluster Density ({dim}Magnitude-Weighted)')

    if 0< len(cluster.iloc[:max_magnitude_index]):
        if len(cluster.iloc[:max_magnitude_index]) <=5:
            foredegree = 1
        elif len(cluster.iloc[:max_magnitude_index]) <= 25:
            foredegree = 2
        elif len(cluster.iloc[:max_magnitude_index]) <= 50:
            foredegree = 3
        elif len(cluster.iloc[:max_magnitude_index]) <= 75:
            foredegree = 4
        elif len(cluster.iloc[:max_magnitude_index]) <= 100:
            foredegree = 5

        try:
            coeffs_fore = np.polyfit(cluster.iloc[:max_magnitude_index+1].delta_days, cluster.iloc[:max_magnitude_index+1].mag, foredegree)
            polynomial_fore = np.poly1d(coeffs_fore)

            x_fit_fore = np.linspace(cluster.iloc[:max_magnitude_index+1].delta_days.min(), cluster.iloc[:max_magnitude_index+1].delta_days.max(), int(max_delta_day)+1)
            y_fit_fore = polynomial_fore(x_fit_fore)

            ax2.plot(x_fit_fore, y_fit_fore, color='red', label=f'Foreshocks curve of degree {foredegree}')
        except : pass

    if 0<len(aftershocks):
        if len(aftershocks) <=3:
            afterdegree = 1
        elif len(aftershocks)/1.5<= afterdegree:
            afterdegree = len(aftershocks)
        try:
            coeffs_after = np.polyfit(cluster.iloc[max_magnitude_index:].delta_days, cluster.iloc[max_magnitude_index:].mag, afterdegree)
            polynomial_after = np.poly1d(coeffs_after)

            x_fit_after = np.linspace(cluster.iloc[max_magnitude_index:].delta_days.min(), cluster.iloc[max_magnitude_index:].delta_days.max(), int(max_delta_day)+1)
            y_fit_after = polynomial_after(x_fit_after)

            ax2.plot(x_fit_after, y_fit_after, color='blue', label=f'Aftershocks of degree {afterdegree}')
        except:pass

    cmapm = plt.get_cmap('RdYlGn_r')
    normm = mcolors.Normalize(vmin=3., vmax=9.2)
    colors = cmapm(normm(cluster.mag))

    ax2.vlines(x = main_earthquake.delta_days, ymin = 0, ymax = 10, linestyles = 'dashed', color = 'grey', alpha = 0.1)

    ax2.scatter(cluster.delta_days, cluster.mag,
                s=cluster['marker_size'],
                c=colors,
                alpha=1,
                label='Evolution of mag')

    ax2.hlines(y=mag_med, xmin=0, xmax=max_delta_day, linestyles='dashed', color='y', label='Aftershock Median')


    smm = ScalarMappable(cmap=cmapm, norm=normm)
    smm.set_array([])
    plt.colorbar(smm, ax=ax2,label='Magnitude Difference from mag_max', orientation='vertical')


    ax2.set_xlim(cluster.iloc[0].delta_days-5, max_delta_day)
    if len(cluster.iloc[:max_magnitude_index]) == 0 and max_delta_day<15:        #to avoid big blank part in the plot
        ax2.set_xlim(cluster.iloc[0].delta_days, max_delta_day)
    elif len(cluster.iloc[:max_magnitude_index]) < 2 and max_delta_day<45:        #to avoid big blank part in the plot
        ax2.set_xlim(cluster.iloc[0].delta_days-2, max_delta_day)
    ax2.set_ylim(cluster.mag.min() - 0.25, cluster.mag.max() + 0.25)
    ax2.set_xlabel('Days since Mainshock')
    ax2.set_ylabel('Magnitude')   # Use set_ylabel for y-axis label

    ax2.legend()
    ax2.grid(True)
    ax2.set_title(f'Magnitude evolution over {int(max_delta_day)} days after mainshock')


    foreshock_text, aftershock_text = shock_text(cluster, aftershocks,max_magnitude_index, mag_med)
    plt.suptitle(f'Near {near_area}, from {date_first_shock} to {date_last_as}\n\nMainshock of Mag {mag_main} reached on {date_ms}\n\n{foreshock_text}, {aftershock_text}', fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_cells_big_earthquake(earthquake_density,big_shock_unique_tuples,grid,world,min_mag,equake_areas):
    # we want to display cells containing at least one big earthquake
    fig, ax = plt.subplots(figsize=(15, 12))
    world.boundary.plot(ax=ax, color='black')

    eqs_grid_cells = {(i, j): index for index, (i, j) in enumerate(earthquake_density['grid_cell'].tolist())}

    for tuple_ij in big_shock_unique_tuples:
        i = tuple_ij[0] - 1
        j = tuple_ij[1] - 1

        min_lat = grid[i, j, 0]
        min_lon = grid[i, j, 1]
        max_lat = grid[i, j, 2]
        max_lon = grid[i, j, 3]

        if (i, j) in eqs_grid_cells:
            earthquake_count = earthquake_density.at[eqs_grid_cells[(i, j)], 'earthquake_count']
            ax.add_patch(plt.Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat, fill=False, color='orange', alpha=0.3))
            equake_areas[i, j] = np.log10(earthquake_count) if earthquake_count != 0 else 0

    ax.scatter(-3.19648,55.95206,c='r', marker="x", label=f" Edinburgh ")
    plt.title(f"Cells containing at least one earthquake of mag {min_mag}")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.xlim(-180,180)
    plt.ylim(-90,90)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_DFS_Kmeans_grid(megaclusters_data,grid,world):
    fig, ax = plt.subplots(figsize=(15, 12))
    world.boundary.plot(ax=ax, color='black')
    max_size = 0
    for megacluster_data in megaclusters_data.values():
        if len(megacluster_data)>max_size:
            max_size = len(megacluster_data)

    for megacluster_id, megacluster_data in megaclusters_data.items():
        colormap = plt.cm.get_cmap('viridis', len(megacluster_data))
        for color_idx,subcluster_cells in enumerate(megacluster_data.values()):
            for cell_tuple in subcluster_cells:
                i, j = cell_tuple

                min_lat = grid[i - 1, j - 1, 0]
                min_lon = grid[i - 1, j - 1, 1]
                max_lat = grid[i - 1, j - 1, 2]
                max_lon = grid[i - 1, j - 1, 3]
                # if len(megacluster_data) <= 20:
                #     color = plt.cm.tab10(color_idx % 10)
                # else:
                color = colormap(color_idx)

                ax.add_patch(plt.Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat, fill=False, color=color))

    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.xlim(-180,180)
    plt.ylim(-90,90)
    plt.title("K-Means clustering over megaclusters of cells (after DFS)")
    plt.tight_layout()
    plt.show()
