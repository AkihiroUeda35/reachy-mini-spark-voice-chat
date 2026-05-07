from reachy_mini import ReachyMini
import numpy as np
# Connect to the running daemon
with ReachyMini() as mini:
    print("Connected to Reachy Mini! ")
    print("Turning on motors...")
    mini.enable_motors()
    mini.wake_up()
    mini.goto_target(head=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), body_yaw=0.5, duration=1)    
    # Wiggle antennas
    print("Wiggling antennas...")
    for i in range(3):
         mini.goto_target(antennas=[0.5, -0.5], duration=1)
         mini.goto_target(antennas=[-0.5, 0.5], duration=1)
    mini.goto_target(antennas=[0, 0], duration=1)
    print("Going to sleep...")
    mini.goto_sleep()
    print("Done!")