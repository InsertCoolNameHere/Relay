import socket  # for socket
import sys
import my_que.my_queue_reader as mq
import threading

jobs_kv = {}
lock = threading.Lock()


# THIS READS IN ITEMS FROM THE QUEUE
class QueueReader(threading.Thread):
    def __init__(self):
        super().__init__()
        redis_host = "redis"
        self.q = mq.RedisWQ(name="gis_q", host=redis_host)

    def run(self):
        self.read_job_queue()

    def read_job_queue(self):
        i = 0
        print("RUNNING INSIDE THE SERVER...")
        while True:
            if not self.q.empty():
                lock.acquire()

                # lease_secs : expiry time after which some other container
                # can pick up the object
                try:
                    item = self.q.lease(lease_secs=300, block=False)

                    if item is not None:
                        gis_code = item.decode("utf-8")
                        print("EXTRACTED:", gis_code)

                        if "$" in gis_code:
                            tokens = gis_code.splie("$")
                            child = tokens[0]
                            parent = tokens[1]
                            if parent not in jobs_kv:
                                jobs_kv[parent] = []
                            jobs_kv[parent].append(child)

                        else:
                            if 'default' not in jobs_kv:
                                jobs_kv['default'] = []
                            jobs_kv['default'].append(gis_code)

                        self.q.complete(item)
                    else:
                        print("Waiting for work")
                finally:
                    lock.release()
            else:
                i += 1
                if i % 500 == 0:
                    print("EMPTY QUEUE...NOTHING TO DO")


# THIS ACCEPTS REQUESTS FROM WORKERS FOR NEW JOBS
class JobScheduler(threading.Thread):
    def __init__(self):
        super().__init__()
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print("Socket successfully created")
        except socket.error as err:
            print("Socket creation failed with error %s" % (err))

        # default port for socket
        port = 31477

        self.s.bind(('', port))
        print ("Socket binded to %s" %(port))

        queue_size = 10
        self.s.listen(queue_size)
        print ("Socket has started listening....")

    def run(self):
        self.job_request_listner()

    def job_request_listner(self):

        while True:
            # Establish connection with client.
            c, addr = self.s.accept()
            print('Got connection from', addr)

            data_from_worker = c.recv(1024).decode()
            print("RECEIVED: ", data_from_worker)

            try:
                lock.acquire()

                gis_work = ""
                # IF THE WORKER HAS NO PARENT (TL PHASE)/ THIS IS AN EXHAUSTIVE TRAINING PHASE
                if "default" in data_from_worker:
                    # POP FROM RELEVANT QUEUE
                    gis_work = jobs_kv['default'].pop(0)
                    # send a thank you message to the client.
                else:
                    # CHECK IF THIS QUEUE HAS ELEMENTS IN IT
                    if len(jobs_kv[data_from_worker]) > 0:
                        gis_work = jobs_kv[data_from_worker].pop(0)

                        # RESPOND WITH CHILD$PARENT
                        gis_work = gis_work+"$"+data_from_worker
                    else:
                        # FIND THE CLUSTER THAT IS THE MOST BACKED UP
                        max_key = max(jobs_kv, key=lambda x: len(set(jobs_kv[x])))
                        gis_work = jobs_kv[max_key].pop(0)

                        # RESPOND WITH CHILD$PARENT
                        gis_work = gis_work + "$" + max_key
                
                gis_work = "Thanks for Connecting"
                c.send(gis_work.encode())
            finally:
                lock.release()


            # Close the connection with the client
            c.close()

if __name__ == '__main__':
    q = QueueReader()
    j = JobScheduler()

    q.start()
    j.start()

    q.join()
    j.join()

    print("ENDING THE PROGRAM....")
