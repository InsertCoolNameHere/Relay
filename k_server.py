import socket
import threading
from utilz.helper import fancy_logging
from cluster.detect_cluster import cluster_setup
import pandas as pd
# THE SERVER RUNS OUTSIDE THE K8S CLUSTER....FOR SIMPLICITY
# CAN BE PUT IN THE CLUSTER, TOO


node_ghash_map = {}
qhash_node_map = {}
parent_inner_qhash_map = {}
node_centroid_map = {}

lock = threading.Lock()

# THIS READS IN ITEMS INTO THE QUEUE
# INPUT IS A FILENAME WITH INDIVIDUAL JOB NAMES
# INPUT FILENAME IS SENT THROUGH k_input_client.py
class ClusterReader(threading.Thread):
    def __init__(self):
        super().__init__()
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            fancy_logging('Input Listener Socket successfully created')
        except socket.error as err:
            print("Input Listener Socket creation failed with error %s" % (err))

        # default port for socket
        file_input_port = 31478

        self.s.bind(('', file_input_port))
        fancy_logging("Input Listener Socket binded to %s" % (file_input_port))
        queue_size = 100
        self.s.listen(queue_size)
        fancy_logging("Input Listener Socket has started listening....")

    def run(self):
        self.read_input_queue()

    def read_input_queue(self):

        # GETS A FILENAME, READS THAT FILE AND BUILDS A CLUSTER OUT OF IT
        while True:
            # Establish connection with client.
            c, addr = self.s.accept()
            fancy_logging('New input data from %s'% (str(addr)))

            data_from_worker = c.recv(1024).decode()
            fancy_logging("FILE INPUT: %s"% str(data_from_worker))

            try:
                lock.acquire()

                node_ghash_map, qhash_node_map, parent_inner_qhash_map, node_centroid_map = cluster_setup(data_from_worker)
            finally:
                lock.release()

            rsp = "Input File " + data_from_worker + " Ingested"
            c.send(rsp.encode())
            # Close the connection with the client
            c.close()
            fancy_logging("CURRENT QUEUES STATE: \n%s\n%s\n%s\n%s"
                          %(str(node_ghash_map), str(qhash_node_map), str(parent_inner_qhash_map), str(node_centroid_map)))

#==============================================================================================================================================================
# THIS ACCEPTS AND GRANTS REQUESTS FROM WORKERS FOR NEW JOBS
class JobScheduler(threading.Thread):
    def __init__(self):
        super().__init__()
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            fancy_logging("Socket successfully created")
        except socket.error as err:
            fancy_logging("Socket creation failed with error %s" % (err))

        # default port for socket
        port = 31477

        self.s.bind(('', port))
        fancy_logging("Socket binded to %s" % (port))

        queue_size = 100
        self.s.listen(queue_size)
        fancy_logging("Socket has started listening....")

    def run(self):
        self.job_request_listner()

    def empty_all_dictionaries(self, requested_parent):
        if (requested_parent in jobs_kv) and len(jobs_kv[requested_parent]) == 0:
            jobs_kv.pop(requested_parent)
            if requested_parent in jobs_load:
                jobs_load.pop(requested_parent)

    # SEE IF THIS NODE HAS ANY CENTROID MODELS THAT ARE WAITING TO BE TRAINED
    def any_centroid_left_tobe_assigned(self, worker_hostname):
        centroid_to_train = None
        if worker_hostname in node_centroid_map:
            candidate_centroids = node_centroid_map[worker_hostname]
            if len(candidate_centroids) > 0:
                centroid_to_train = candidate_centroids.pop()
        return centroid_to_train

    def job_request_listner(self):

        while True:
            # Establish connection with client.
            c, addr = self.s.accept()
            fancy_logging('GOT A CONNECTION FROM %s'% str(addr))

            request_from_worker = c.recv(1024).decode()
            fancy_logging("REQUEST RECEIVED: %s"% str(request_from_worker))

            # EXTRACT WORKER_ID AND GIS_CODES
            if "$" in request_from_worker:
                tokens = request_from_worker.split("$")
                requested_parent = tokens[0]
                worker_hostname = tokens[1]

                if "default" in requested_parent:
                    old_centroid = "default"
                else:
                    old_centroid = requested_parent[0:8]
            else:
                fancy_logging("ERROR!!!! RECEIVED: %s"% request_from_worker)
                return "ERROR-TRYAGAIN"

            # WORK ASSIGNMENT BASED ON WORKER AND GIS_CODES
            try:
                lock.acquire()

                gis_work = ""
                further_looking = True

                # IF THIS NODE HAS CENTROID MODELS LEFT TO BE TRAINED, ASSIGN THEM FIRST
                centroid_to_train = self.any_centroid_left_tobe_assigned(worker_hostname)

                if centroid_to_train:
                    # IF THIS NODE HAS SOME CENTROID MODELS LEFT TO BE EXHAUSTIVELY TRAINED
                    gis_work = centroid_to_train
                else:
                    print("All Centroids have been trained/being trained")
                    # IF OLD_CENTROID IS NOT 'default'




                # TRANSFER LEARNING PHASE
                if further_looking:
                    # IF THIS PARENT HAS REMAINING CHILDREN...SIMPLY POP THE NEXT ITEM AND HAND IT OVER
                    # IF THIS PARENT HAS ONLY ONE CHILDREN LEFT, ALSO REMOVE ITS TRACES FROM ALL DICTIONARIES
                    if (requested_parent in jobs_kv) and len(jobs_kv[requested_parent]) > 0:
                        gis_work = jobs_kv[requested_parent].pop(0)

                        # RESPOND WITH CHILD$PARENT
                        gis_work = gis_work + "$" + requested_parent

                        # IF THIS IS A NEWLY ASSIGNED WORKER FOR THIS PARENT, PUT IT IN THE MAP
                        # KEEPING TRACK OF UNIQUE WORKERS UNDER EACH PARENT
                        if requested_parent not in jobs_load:
                            jobs_load[requested_parent] = []

                        if worker_hostname not in jobs_load[requested_parent]:
                            jobs_load[requested_parent].append(worker_hostname)

                        # THIS IS THE LAST JOB FOR THIS PARENT
                        # REMOVE IT FROM THE KV-PAIR AND LOAD QUEUE
                        self.empty_all_dictionaries(requested_parent)

                    # IF THIS PARENT HAS NO CHILDREN LEFT
                    # BUT THE QUEUE MIGHT HAVE OTHER LAGGING PARENTS
                    elif len(jobs_kv) > 0:

                        #print("\n\nASSIGNING YOU NEW PARENT.......\n\n")
                        # FIND THE CLUSTER THAT IS THE MOST BACKED UP
                        # THIS MIGHT NEED A SELF-WRITTEN FUNCTION
                        #print(jobs_kv,jobs_load)

                        # FIND PARENTS WITH JOBS THAT HAVE NO WORKERS HANDLING THEM
                        parents = list(filter(lambda x: len(set(jobs_kv[x]))>0 and (x not in jobs_load or len(jobs_load[x]) == 0), jobs_kv))

                        #print("POSSIBLE PARENTS: ", parents)
                        if len(parents) > 0:
                            new_parent = parents[0]
                        else:
                        # FIND THE PARENTS THAT ARE MOST BACKED UP
                            new_parent = max(jobs_kv, key=lambda x: len(set(jobs_kv[x]))/ len(set(jobs_load[x])))

                        # WHAT ABOUT INF

                        if new_parent is None:
                            # AN UNLIKELY SCENARIO
                            gis_work = "TRYAGAIN"
                        else:
                            gis_work = jobs_kv[new_parent].pop(0)

                            # RESPOND WITH CHILD$PARENT
                            gis_work = gis_work + "$" + new_parent

                            # ASSIGNING THE WORKER TO THIS NEW PARENT
                            if new_parent not in jobs_load:
                                jobs_load[new_parent] = []

                            if worker_hostname not in jobs_load[new_parent]:
                                jobs_load[new_parent].append(worker_hostname)
                        
                        # IF THIS PARENT HAD ONLY 1 JOB IN IT,
                        # REMOVE ALL TRACES OF THIS PARENT FROM DICTIONARIES
                        self.empty_all_dictionaries(new_parent)

                    else:
                        gis_work="TRYAGAIN"
                    # THIS NEEDS A TRY_AGAIN

                fancy_logging("RESPONDED WITH: %s"% str(gis_work))
                #print("\n\nCURRENT STATE: ", jobs_kv, jobs_load, "\n\n")
                c.send(gis_work.encode())
            finally:
                lock.release()

            # Close the connection with the client
            c.close()

if __name__ == '__main1__':
    jobs_kv = {"riki":[1,2,3], "a":[1,2]}
    jobs_load = {"riki":[1]}
    new_parent = list(filter(lambda x: len(set(jobs_kv[x]))>0 and (x not in jobs_load or len(jobs_load[x]) == 0), jobs_kv))

    print(new_parent)

if __name__ == '__main__':

    cluster_setup()
    exit(1)
    q = ClusterReader()
    j = JobScheduler()

    q.start()
    j.start()

    q.join()
    j.join()

    fancy_logging("ENDING THE PROGRAM....")
