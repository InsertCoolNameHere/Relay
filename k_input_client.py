import socket
import sys


# SENDS A FILE NAME TO THE SERVER WITH NODE NAMES
def send_input_file_req(str):
    s.send(str.encode())
    data = s.recv(1024).decode()
    print("RECEIVED RESPONSE:", data)

# EXAMPLE
# python3.6 k_input_client FILE_PATH
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("NEED INPUT FILENAME...TRY AGAIN")
        exit(1)

    filename = sys.argv[1]

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "kiwis.cs.colostate.edu"
    file_input_port = 31478

    s.connect((host, file_input_port))
    print("CONNECTION SUCCESSFUL")
    send_input_file_req(filename)
    s.close()

