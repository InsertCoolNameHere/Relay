import math
import pickle
import os
import socket
parent_len = 8

lattice_low = 177
lattice_high = 191
def group_qhashes(nodes_file):
    # FIRST RUN /s/chopin/e/proj/sustain/sapmitra/mongodb/mongo_ingest_glance/scripts/distributed_find_quadhashes.sh lattices_small > nodes.csv
    nodes_file = open(nodes_file, 'r')
    Lines = nodes_file.readlines()

    node_ghash_map = {}
    qhash_node_map = {}
    parent_inner_qhash_map = {}
    current_node = ""
    current_qhashes = []

    for l in Lines:
        line = l.strip()
        if "lattice" in line:
            if len(current_qhashes) > 0:
                copied_qhashes = current_qhashes.copy()
                node_ghash_map[current_node] = copied_qhashes
                current_qhashes = []
            current_node = line
        else:
            tokens = line.split(',')

            cnt = int(tokens[0])
            line = tokens[1]

            if cnt >=2:
                current_qhashes.append(line)

                if line in qhash_node_map:
                    print("DUPLICATE NODE FOR ", line)
                else:
                    qhash_node_map[line] = current_node
                    parent_qhash = line[0:8]

                    if parent_qhash in parent_inner_qhash_map:
                        qhash_arr = parent_inner_qhash_map[parent_qhash]['inner']
                    else:
                        qhash_dict = {}
                        qhash_arr = []
                        qhash_dict['inner'] = qhash_arr
                        parent_inner_qhash_map[parent_qhash] = qhash_dict
                    qhash_arr.append(line)

    if len(current_qhashes) > 0:
        copied_qhashes = current_qhashes.copy()
        node_ghash_map[current_node] = copied_qhashes

    return node_ghash_map, qhash_node_map, parent_inner_qhash_map

def coordinate_in_grid(quad_str):

    quad_substr  = quad_str[parent_len:]
    x = 0
    y = 0

    wid = math.pow(2, len(quad_substr)-1)

    for s in quad_substr:
        if s == '0':
            x+=0
        elif s == '1':
            x += wid
        elif s == '2':
            y += wid
        elif s == '3':
            x += wid
            y += wid
        wid/=2

    dist = math.pow(x-3.5, 2) + math.pow(y-3.5, 2)
    #print(dist)
    return dist

def populate_node_centroid_map(centroid_hash, node_centroid_map, qhash_node_map):
    node_name = qhash_node_map[centroid_hash]
    if node_name in node_centroid_map:
        arr = node_centroid_map[node_name]
    else:
        arr = []
        node_centroid_map[node_name] = arr
    arr.append({'centroid':centroid_hash, 'trained': False})

def cluster_setup(nodes_file):

    node_ghash_map,qhash_node_map, parent_inner_qhash_map = group_qhashes(nodes_file)
    node_centroid_map = {}
    #parent_inner_qhash_map = {'02310102':{'inner':['02310102033', '02310102122', '02310102211', '02310102000']}}

    # FINDING THE CENTROID OF EACH CLUSTER
    for parent_hash in parent_inner_qhash_map:
        inner_dict = parent_inner_qhash_map[parent_hash]

        inner_quadhashes = inner_dict['inner']

        if len(inner_quadhashes) == 1:
            centroid_qhash = inner_quadhashes[0]
            populate_node_centroid_map(centroid_qhash, node_centroid_map, qhash_node_map)
            inner_dict['centroid'] = centroid_qhash
            inner_quadhashes.remove(centroid_qhash)
            continue

        # GET THE QUADHASH CLOSEST TO THE CENTER OF THIS CLUSTER
        cent = min(inner_quadhashes, key=coordinate_in_grid)
        populate_node_centroid_map(cent, node_centroid_map, qhash_node_map)
        inner_dict['centroid'] = cent
        inner_quadhashes.remove(cent)

    print(node_ghash_map)
    print(len(qhash_node_map), qhash_node_map)
    print(parent_inner_qhash_map)
    print(node_centroid_map)

    return node_ghash_map, qhash_node_map, parent_inner_qhash_map, node_centroid_map

def cluster_setup_nlcd(tif_path, centroids, all_nodes):
    # ALL QUADHASHES ON THIS NODE
    all_qh_here = []
    new_map = {}
    # FIND ALL QUADHASHES ON THIS NODE
    for file in os.listdir(tif_path):
        all_qh_here.append(str(file))

    #print(all_qh_here)
    parents = []
    with open(centroids, 'rb') as fp:
        centroid_map = pickle.load(fp)
    with open(all_nodes, 'rb') as fp:
        nodes_map = pickle.load(fp)

    for key in centroid_map:
        print(type(centroid_map[key]), centroid_map[key])
        if centroid_map[key] in all_qh_here:
            parents.append(key)
    #print("PARENTS", parents)

    for key_p in parents:
        parent_qh = centroid_map[key_p]
        children = nodes_map[key_p]
        here_children = []
        for c in children:
            if c in all_qh_here and c not in parent_qh:
                here_children.append(c)

        if len(here_children) > 0:
            new_map[parent_qh] = here_children

    #print(new_map)
    return new_map



if __name__ == '__main__':
    my_hostname = str(socket.gethostname())
    img_dir = "/s/" + my_hostname + "/a/nobackup/galileo/stip-images/co-3month/Sentinel-2/"
    cluster_setup_nlcd(img_dir, "mapz/cluster_centroids", "mapz/cluster_lite")








if __name__ == '__main1__':
    node_ghash_map, qhash_node_map, parent_inner_qhash_map, node_centroid_map = cluster_setup('nodes.csv')

    print(node_ghash_map)
    print("===========================================================")
    print(qhash_node_map)
    print("===========================================================")
    print(parent_inner_qhash_map)
    print("===========================================================")
    print(node_centroid_map)

    for k in parent_inner_qhash_map.keys():
        inners = parent_inner_qhash_map[k]['inner']
        centroid = parent_inner_qhash_map[k]['centroid']
        print(centroid, len(inners))






