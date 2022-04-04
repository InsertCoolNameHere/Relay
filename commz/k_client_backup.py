import socket


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "kiwis.cs.colostate.edu"
port = 31477
s.connect((host,port))

def ts(str):
   s.send(str.encode())
   data = s.recv(1024).decode()
   print("RESPONSE:", data)

ts("HELLO I AM KIWIS")

s.close ()