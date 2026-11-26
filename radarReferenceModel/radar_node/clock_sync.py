"""
Set the PYNQ's clock from the Mac to within a few milliseconds (run on the Mac).

    python3 clock_sync.py                 # board at xilinx@192.168.2.99

Why not `ssh ... date -s $(date)`: the Mac's time is read BEFORE ssh connects, so the
connection time (and any password typing) ends up as a clock error on the board.
Here the board says "go" once it is connected, and only then is the Mac's time sent.
Afterwards the remaining offset is measured (board time vs. the midpoint of a round trip).
Needs key login (run `ssh-copy-id xilinx@192.168.2.99` once).
"""
import subprocess
import sys
import time

HOST = sys.argv[1] if len(sys.argv) > 1 else "xilinx@192.168.2.99"
PW = "xilinx"   # PYNQ image default sudo password

# sudo starts first (about 1.3 s on the A9), THEN the board says go and the Mac sends its time
remote = 'sudo -S -p "" sh -c \'echo go; read t; date -u -s "@$t" > /dev/null && echo set\''
p = subprocess.Popen(["ssh", HOST, remote], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
p.stdin.write(PW + "\n")
p.stdin.flush()
if p.stdout.readline().strip() != "go":
    sys.exit("no answer from the board (is it on, and does `ssh " + HOST + "` work without a password?)")
p.stdin.write(f"{time.time():.3f}\n")
p.stdin.flush()
out = p.communicate(timeout=20)[0]
if "set" not in out:
    sys.exit("could not set the clock: " + out)

# measure what is left: ask for the board time again over one connection
p = subprocess.Popen(["ssh", HOST, "read x; date +%s.%N"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
time.sleep(1.5)                      # let ssh finish connecting
t0 = time.time()
p.stdin.write("x\n")
p.stdin.flush()
board = float(p.stdout.readline())
t1 = time.time()
p.wait()
print(f"board clock set; remaining offset {(board - (t0 + t1) / 2) * 1e3:+.0f} ms "
      f"(measured over a {(t1 - t0) * 1e3:.0f} ms round trip)")
