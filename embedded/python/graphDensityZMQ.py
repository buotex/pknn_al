import math
import numpy 
import json
import zmq
import graphdensity

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    socket.send(A, flags, copy=copy, track=track)
    return socket.recv()

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    s.send("OK")
    buf = buffer(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])


ctx = zmq.Context()
s = ctx.socket(zmq.REP)
s.bind("ipc:///tmp/zmq-test")
pref = s.recv()
s.send("OK")
X = recv_array(s)
##
##
while True:
    density = graphdensity.Density(X)
    while True:
        pickedIndex = s.recv()
        if pickedIndex == "reset":
            s.send("OK")
            break
        density.pick(int(pickedIndex))
        send_array(s,density.getDensity())
        s.send("OK")
#plt.clf()
