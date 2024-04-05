import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from other_functions import haversine

def get_adjacent_cells(cell,cell_index_map):
    """
    Args:
    - cell (tuple) : the cell tuple containing earthquakes we want to cluster
    - cells_with_data (list of tuples) : the list of all the cells with earthquakes inside
    Return :
    - adjacent_cells_with_data (list) : the adjacent cells of the focused cell only if there is data to cluster inside
        size of the list returned : min = 0, max = 8
    """
    x, y = cell
    adjacent_cells = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                      (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)]
    adjacent_cells_with_data = [adj_cell for adj_cell in adjacent_cells if adj_cell in cell_index_map]

    return adjacent_cells_with_data


def iterative_dfs(start_cell,cell_index_map,visited,num_latitude_cells,num_longitude_cells):
    stack = [start_cell]
    cluster = []

    while stack:
        cell = stack.pop()
        if cell not in visited:
            visited.add(cell)
            cluster.append(cell)
            neighbors = get_adjacent_cells(cell, cell_index_map)
            temp = []
            if cell[0] == 0:
                temp = [(num_latitude_cells, cell[1]-1),(num_latitude_cells, cell[1]),(num_latitude_cells, cell[1]+1)]
            if cell[1] == 0:
                temp = [(cell[0] - 1, num_longitude_cells),(cell[0],num_longitude_cells),(cell[0] + 1,num_longitude_cells)]
            if cell[0] == num_latitude_cells:
                temp = [(0, cell[1]-1),(0, cell[1]),(0, cell[1]+1)]
            if cell[1] == num_longitude_cells:
                temp = [(cell[0] - 1, 0),(cell[0],0),(cell[0] + 1,0)]
            if len(temp) > 0:
                temp = [adj_cell for adj_cell in temp if adj_cell in cell_index_map]
                neighbors += temp

            for neighbor in neighbors:
                neighbor_index = cell_index_map.get(neighbor)

                if neighbor_index is not None and neighbor not in visited:
                    stack.append(neighbor)

    return cluster

def create_cells_chain_clusters(big_earthquake_db, no_bigshock_cells, cells, cell_index_map, num_latitude_cells,num_longitude_cells, large_cell_cluster_size,k_plus):

    visited = set()
    adjacent_shocks_cells = []
    isolated_shocks_cells = []
    score_list = []
    megaclusters_labels= {}
    megaclusters_data = {}

    def cluster_large_cluster(megacluster):
        """
        In case of a large cluster of cell (megacluster), we divide the megacluster into smaller subclusters.
        this is done with K-means, with the number of centroids 'k' defined thanks to 'large_cell_cluster_size'
        We can tune (adjust) the parameter 'k_plus' in order to create more or less subclusters.
        For good results, a 'k_plus' value between -2 and 2 is recommended after several experimentations.

        Splitting a megacluster intosubclusters can lead to an issue : what if we spatially split a cluster (of earthquakes) into pieces?
        This can happen in the case of very large finit fault and aftershocks that are distant from the mainshock. Then, we
        need to keep track of the different adjacent cluster in order to rebuild the splitted clusters later.
        """
        subclusters_adjency = {}
        subclusters_data = {}
        cell_cluster_label_map = {}

        k = int(len(megacluster) / large_cell_cluster_size)+ k_plus
        if k<= 1:
            k =2 #cannot perform Kmeans if value <= 1

        cluster_coords = [[cell[0], cell[1]] for cell in megacluster]

        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)    #the other parameters are set to default values
        kmeans.fit(cluster_coords)
        cluster_labels = kmeans.labels_
        sil_score = silhouette_score(cluster_coords, cluster_labels)    #used to determine the quality of our k-means clustering
        score_list.append(sil_score)

        subclusters = {i: [] for i in range(k)}

        for i, cell in enumerate(megacluster):
            subcluster_index = cluster_labels[i]
            subclusters[subcluster_index].append(cell)
            cell_cluster_label_map[cell] = subcluster_index

        for subcluster_label, subcluster_data in subclusters.items():
            adjacent_clusters = set()

            for cell in subcluster_data:
                adjacent_cells = get_adjacent_cells(cell,cell_index_map)
                for adjacent_cell in adjacent_cells:
                    if cell_cluster_label_map[adjacent_cell] != subcluster_label:
                        adjacent_clusters.add(cell_cluster_label_map[adjacent_cell])

            # add_cluster(subcluster_data)

            subclusters_adjency[subcluster_label] = adjacent_clusters
            subclusters_data[subcluster_label] = subcluster_data

        megaclusters_labels[len(megaclusters_labels)] = subclusters_adjency
        megaclusters_data[len(megaclusters_data)] = subclusters_data

        print(f"Megacluster ({len(megaclusters_labels)}) Silhouette Coefficient with Noise as one cluster: {sil_score:.3f}")

    def add_cluster(new_cluster):
        if len(new_cluster) > 1 :
            #keep the next line or not??? If we keep the line, we keep the cluster if and only if there is at least one big shock in the cluster
            #as we are working on cluster of cells of reasonable size (not a megacluster), then yes we need at least one big shock in this area
            if len(big_earthquake_db[big_earthquake_db['grid_cell'].isin(new_cluster)]) > 0: #to keep only clusters with bigshocks, otherwise no study interest for us
                adjacent_shocks_cells.append(new_cluster)
        else:
            if cell not in no_bigshock_cells: #to keep only clusters with bigshocks, otherwise no study interest for us
                isolated_shocks_cells.append(new_cluster[0])

    for cell in cells:
        if cell not in visited:# and len(megaclusters_data)==0:
            new_cluster = iterative_dfs(cell,cell_index_map,visited,num_latitude_cells,num_longitude_cells)

            if len(new_cluster)>=large_cell_cluster_size:
                cluster_large_cluster(new_cluster)
            else:
                add_cluster(new_cluster)


    print(f"Over {len(megaclusters_labels)} megaclusters : silhouette score = {np.mean(score_list)}")

    return megaclusters_labels, megaclusters_data, adjacent_shocks_cells, isolated_shocks_cells


def filter_clusters(cluster_data):
    cluster_max_mag = cluster_data['mag'].max()
    return cluster_max_mag >= 5.

def dbscan_temporal(all_shocks,eps_value):
    """
    Considering we already pre-clustered the cells accurately
    thanks to the iterative DFS combined to the K-means,
    we only have to consider temporal events, to avoid
    working on a limited spatial distance.

    This gives from far the best results.
    It's very accurate and also time efficient.
    """
    max_mag = all_shocks['mag'].max()
    epsilon = eps_value * max_mag**0.95   #proportionnal value of epsilon because big earthquakes generally have aftershocks more spread over time

    X = all_shocks['timestamp'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=epsilon, min_samples=1, n_jobs=-1)

    labels = dbscan.fit_predict(X_scaled)

    all_shocks['cluster'] = labels

    clusters = all_shocks.groupby('cluster').filter(filter_clusters)

    return clusters

def select_rows(data):
    main_earthquake = data.loc[data['mag'].idxmax()]
    try:
        return main_earthquake, data[(data['id'] != main_earthquake['id'])].copy().drop(['BigShock'], axis = 1)

    except:
        return main_earthquake, data[(data['id'] !=  main_earthquake['id'])].copy()


def update_dist_cluster(main_earthquake, selected_earthquakes):
    """
    updates the distance between the mainshock and the fore/after shocks in the dataframe
    fast computing time
    """
    if not selected_earthquakes.empty:
        selected_earthquakes['delta_depth'] = main_earthquake['depth'] -selected_earthquakes['depth']
        selected_earthquakes['delta_mag']  = main_earthquake['mag'] -selected_earthquakes['mag']

        selected_earthquakes['delta_days'] = (pd.to_datetime(selected_earthquakes['time']) - pd.to_datetime(main_earthquake['time'])).dt.total_seconds() / 86400
        selected_earthquakes['delta_dist'] = selected_earthquakes.apply(
            lambda row: haversine(main_earthquake['latitude'], main_earthquake['longitude'], row['latitude'], row['longitude']), axis=1
        )
        selected_earthquakes = selected_earthquakes[selected_earthquakes['delta_dist']<550]


    main_earthquake_df = pd.DataFrame([main_earthquake])
    main_earthquake_df.loc[:, 'delta_depth'] = 0
    main_earthquake_df.loc[:, 'delta_mag'] = 0
    main_earthquake_df.loc[:, 'delta_days'] = 0
    main_earthquake_df.loc[:, 'delta_dist'] = 0
    try:
        main_earthquake_df = main_earthquake_df.drop(['BigShock'], axis=1)
    except:pass #means we are working on outliers

    cluster_data_earthquakes = pd.concat([main_earthquake_df, selected_earthquakes], ignore_index=True).sort_values('time').reset_index(drop = True)
    try:cluster_data_earthquakes = cluster_data_earthquakes.drop(['cluster'], axis = 1)
    except:pass
    return cluster_data_earthquakes

def process_cluster_data(cluster_data, subcluster_data):
    cluster_data = cluster_data.reset_index(drop=True)

    if cluster_data['mag'].max()<5:
        return pd.DataFrame()

    main_earthquake = cluster_data.loc[cluster_data['mag'].idxmax()]

    if main_earthquake.grid_cell not in subcluster_data:
        return pd.DataFrame()

    main_earthquake, selected_rows  = select_rows(cluster_data)

    cluster_data_earthquakes = update_dist_cluster(main_earthquake, selected_rows)
    return cluster_data_earthquakes

def find_outliers(cluster_data_earthquakes, st_cluster_size_threshold = 100):
    """
    detects the outliers according to 2 strategies in function of the size of the dataframe.
    if size < 100 (foreshocks included) it computes the pair 2 pair distance between each shock
    else, it applies a gaussian filter as : mean + 2*sigma (value can be adapted)

    over st_cluster_size_threshold = 100, we have quite a big "n" value,
    then we can say that the distribution follows a Normal law. A 2-sigma rule
    seems appropriate to detect the outliers over this threshold.
    (we use 2.25 and not 2 to encapsulate high variablity of natural hazards)

    fast computing time
    """

    outliers_df = pd.DataFrame(columns=cluster_data_earthquakes.columns)
    small_outliers_df = pd.DataFrame(columns=cluster_data_earthquakes.columns)
    nb_sigma = 2.25

    mag_max_idx = cluster_data_earthquakes.mag.idxmax()
    mag_max = cluster_data_earthquakes.loc[mag_max_idx]['mag']

    if len(cluster_data_earthquakes)>st_cluster_size_threshold:
        dist_std= round(np.std(cluster_data_earthquakes['delta_dist']),2)
        dist_mean = round(np.mean(cluster_data_earthquakes['delta_dist']),2)
        outliers_df = cluster_data_earthquakes[cluster_data_earthquakes['delta_dist'] > (dist_mean + nb_sigma * dist_std)]

    elif len(cluster_data_earthquakes)<=st_cluster_size_threshold:# or len(outliers_df)==0: #lets consider under the threshold it detects only isolated cells
        a = 50
        b = -200
        dist_threshold = a*mag_max+b
        for i in range(len(cluster_data_earthquakes)):
            min_distance = float('inf')
            for j in range(len(cluster_data_earthquakes)):
                if i != j:
                    distance = haversine(cluster_data_earthquakes['latitude'][i], cluster_data_earthquakes['longitude'][i], cluster_data_earthquakes['latitude'][j], cluster_data_earthquakes['longitude'][j])
                    if distance < min_distance:
                        min_distance = distance
            if min_distance >= dist_threshold :  #if the closest distance between one point and its closest neighbor is < dist_threshold : it's an outlier
                if cluster_data_earthquakes.iloc[i]['mag']>=5:
                    outliers_df = outliers_df.append(cluster_data_earthquakes.iloc[i], ignore_index=True)
                else:
                    small_outliers_df = small_outliers_df.append(cluster_data_earthquakes.iloc[i], ignore_index=True)

    cluster_data_earthquakes = cluster_data_earthquakes[~cluster_data_earthquakes['id'].isin(outliers_df['id'].values.tolist()+small_outliers_df['id'].values.tolist())].copy().reset_index(drop = True)

    if len(outliers_df)>0 and len(cluster_data_earthquakes)>0: #we need to update the original cluster because they are less shocks now!
        if cluster_data_earthquakes.mag.max()<5: #if no more
            cluster_data_earthquakes = pd.DataFrame(columns=cluster_data_earthquakes.columns)

        else:
            main_earthquake, selected_rows  = select_rows(cluster_data_earthquakes)
            cluster_data_earthquakes = update_dist_cluster(main_earthquake, selected_rows)

    if outliers_df.mag.max() <5:
        outliers_df = pd.DataFrame(columns=cluster_data_earthquakes.columns)    #we want outliers only if mag >= 5 : doing this
                                                                                #means we filtered outliers from cluster_data_earthquakes
                                                                                #but we will not work on them (they are useless)
    if len(outliers_df) == 1:
        outliers_df = outliers_df.reset_index(drop = True)
        outliers_df['delta_depth'] = 0
        outliers_df['delta_mag'] = 0
        outliers_df['delta_dist'] = 0
        outliers_df['delta_days'] = 0

    return cluster_data_earthquakes, outliers_df

def cluster_outliers(outliers_df, original_cluster_size, num_latitude_cells,num_longitude_cells):
    """
    list all the outliers as dataframes, taking into account the size of the original spatio temporal cluster
    -> if it's now 0, it means that every shocks are distant from each other (isolated according to the p2p distance).
    -> otherwise, we apply a new spatial clustering over the outliers with DBSCAN
    expensive computing time for the new DBSCAN
    """

    outliers_df_list = []
    mag_max = outliers_df['mag'].max()  #we still rely a bit on the magnitude but inside the outlayers

    if mag_max <5:
        return outliers_df_list

    if original_cluster_size ==0 :
        """means that all shocks are very distant from each other by default (any cluster of outliers)"""
        for i,outlier in outliers_df.iterrows():
            if outlier["mag"] >= 5:
                outlier_df = pd.DataFrame([outlier])

                outlier_df['delta_depth'] = 0.  # Update the 'delta_depth' column
                outlier_df['delta_mag']  = 0.  # Update the 'delta_mag' column
                outlier_df['delta_dist'] = 0.  # Update the 'delta_dist' column
                outlier_df['delta_days'] = 0.  # Update the 'delta_days' column

                outliers_df_list.append(outlier_df.reset_index(drop=True))

        return outliers_df_list

    else:

        if len(outliers_df)<=50:    #threshold manually chosen

            visited = set()
            cells_with_data = outliers_df.grid_cell.tolist()
            cell_index_map = {cell: i for i, cell in enumerate(cells_with_data)}

            for cell in cells_with_data:
                if cell not in visited:
                    new_cluster = iterative_dfs(cell,cell_index_map,visited,num_latitude_cells,num_longitude_cells)
                    outlier_df = outliers_df[outliers_df['grid_cell'].isin(new_cluster)]
                    if outlier_df.mag.max()>=5:
                        outlier_df = outlier_df.reset_index(drop=True)
                        main_earthquake, selected_rows  = select_rows(outlier_df)

                        outlier_df = update_dist_cluster(main_earthquake, selected_rows)
                        outliers_df_list.append(outlier_df)
            return outliers_df_list

        else:

            """
            We need to find the possible afinities between the outliers
            Problem: takes too much time to reapply a spatial cluster.
            """

            epsilon = 0.025*mag_max #0.175

            X = outliers_df[['latitude','longitude']]

            #strat 1 (usual)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            dbscan = DBSCAN(eps=epsilon, min_samples=1, n_jobs=-1)
            labels = dbscan.fit_predict(X_scaled)
            outliers_df['cluster'] = labels
            outliers_clusters = outliers_df.groupby('cluster').filter(filter_clusters)

            if len(outliers_clusters)>0:
                for outlier_label, outlier_data in outliers_clusters.groupby('cluster'):

                    outlier_data = outlier_data.reset_index(drop=True)
                    if outlier_data.mag.max()>=5:

                        main_earthquake, selected_rows = select_rows(outlier_data)

                        outlier_data = update_dist_cluster(main_earthquake, selected_rows)
                        outliers_df_list.append(outlier_data)

            return outliers_df_list

def form_temporal_shocks_clusters(all_shocks, eps_value, subcluster_data, num_latitude_cells,num_longitude_cells, high_quality = False):
    area_clusters = dict()

    clusters = dbscan_temporal(all_shocks,eps_value)

    for cluster_label, cluster_data in clusters.groupby('cluster'):
        cluster_data_earthquakes = process_cluster_data(cluster_data, subcluster_data)  #we create our spatio temporal cluster

        if high_quality:
            if len(cluster_data_earthquakes)==1:        #if only one shock, we know we can direclty add the unique shock cluster to the area ST clusters
                add_cluster(area_clusters, cluster_label, cluster_data_earthquakes)

            elif len(cluster_data_earthquakes)>1:   #but if its bigger, there can be outliers
                original_cluster, outliers_df = find_outliers(cluster_data_earthquakes)

                if len(original_cluster)>0: #we add the original cluster filtered from its outliers if not empty
                    add_cluster(area_clusters, cluster_label, original_cluster)


                if len(outliers_df)==1:     #if only one outlier, we can directly add it (delta already set to 0)
                    outlier_cluster_label = f"{cluster_label}_0"
                    # add_cluster(area_clusters, outlier_cluster_label, outliers_df)
                    area_clusters[outlier_cluster_label] = outliers_df

                elif len(outliers_df)>1:    #if more than one outlier, we need to find the possible correlations between outliers
                    additional_clusters = cluster_outliers(outliers_df, len(original_cluster), num_latitude_cells,num_longitude_cells)
                    if len(additional_clusters) > 0:
                        for i,additional_cluster in enumerate(additional_clusters):
                            outlier_cluster_label = f"{cluster_label}_{i}"
                            area_clusters[outlier_cluster_label] = additional_cluster


        else:
            if len(cluster_data_earthquakes)>0:
                add_cluster(area_clusters, cluster_label, cluster_data_earthquakes)
    return area_clusters


def add_cluster(area_clusters, cluster_label, cluster_data):
    main_earthquake = cluster_data.loc[cluster_data['mag'].idxmax()].copy()

    main_earthquake['time'] = pd.to_datetime(main_earthquake['time'])
    current_date = datetime.now(timezone.utc)
    if (current_date - main_earthquake['time']).days >= 30:
        area_clusters[cluster_label] = cluster_data


def cluster_shocks_by_temporal_feature(all_clusters_dict,megaclusters_data,megaclusters_labels,big_earthquake_db,small_earthquakes,num_latitude_cells,num_longitude_cells,subclusters_total, high_quality = True):
    cluster_dict = {}
    area = 0
    eps_value = 2.42e-4 #in this function for the moment because I need to do more test @Ben
                        #in the future it will surely be fixed to a specific constant

    megaclusters_labels_list = list(megaclusters_labels.keys())
    for megacluster_label, megacluster_data in all_clusters_dict.items():
        cluster_dict[megacluster_label] = {}

        for subcluster_label, subcluster_data in megacluster_data.items():
            # if megacluster_label>0 or subcluster_label >0:
            #     continue
            cells_cluster = subcluster_data.copy()
            adj_subclusters = []
            if megacluster_label in megaclusters_labels_list:
                adj_subclusters = megaclusters_labels[megacluster_label][subcluster_label]
                for adj_subcluster_label in adj_subclusters:
                    cells_cluster += megaclusters_data[megacluster_label][adj_subcluster_label]


            major_shocks = big_earthquake_db[big_earthquake_db['grid_cell'].isin(cells_cluster)]
            minor_shocks = small_earthquakes[small_earthquakes['grid_cell'].isin(cells_cluster)]#.drop('normalized_mag', axis = 1)
            all_shocks = pd.concat([major_shocks, minor_shocks], ignore_index=True)
            all_shocks = all_shocks.drop('normalized_mag', axis = 1)

            area_clusters = form_temporal_shocks_clusters(all_shocks, eps_value, subcluster_data,num_latitude_cells,num_longitude_cells, high_quality)

            if len(area_clusters)>0:
                cluster_dict[megacluster_label][subcluster_label] =area_clusters

            if (area)%25 == 0 :
                print(f"area {area}/{subclusters_total} : {round((area+1)/subclusters_total*100,2)}% completed")
            if area+1 == subclusters_total:
                print(f"area {area+1}/{subclusters_total} : {round((area+1)/subclusters_total*100,2)}% completed")

            area += 1
            # print(area)

    return cluster_dict
