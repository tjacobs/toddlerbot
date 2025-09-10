### Notes on downloading the VR unity app

#### Steps
##### Option 1
Download `toddy_ar.apk` file from [link](https://drive.google.com/drive/folders/1GVy6MPUSzoXVRfidI-VVwjzZaCeVIUIF?usp=drive_link) and load it to your VR device. The app is built for Quest 2.
##### Option 2
Download the source code `toddy_ar.zip`, extract the files and copy the folders to your unity project. Select the appropriate platform you want the project to be built on, and build the project. 

#### Usage
1. Enter the IP of your work station where the teleoperation script will be running on, then press `O` to continue.
2. A verification interface will show up to check the correctness of your IP. If you confirm that the IP you entered is correct, press `O` to continue. Or press `A` to return back to step 1.
3. Start your teleoperation!

# Teleoperation README

This guide mirrors the original bash script and explains how to run the teleoperation demos—both with a simulator and with a real robot.

---

## Quick test

```bash
python examples/test_teleoperation_gmr.py
```

---

## Teleoperation + Simulator

Open **two terminals**.

**Terminal 1 — follower (PD)**
```bash
python toddlerbot/policies/run_policy.py --policy teleop_follower_pd --robot toddlerbot_2xc --vis view --task teleop_vr
```

**Terminal 2 — leader (VR)**
```bash
python toddlerbot/policies/run_policy.py --policy teleop_vr_leader --robot toddlerbot_2xc
```

---

## Teleoperation + Real Robot

You’ll run one process on the robot (“toddy”) and one on your desktop.  
Replace the IPs as noted.

**On toddy — follower (PD)**
```bash
python toddlerbot/policies/run_policy.py --policy teleop_follower_pd --robot toddlerbot_2xm --sim real --task teleop_vr --ip 192.168.0.203
```
> Replace `192.168.0.203` with **your desktop’s IP**.

**On your desktop — leader (VR)**
```bash
python toddlerbot/policies/run_policy.py --policy teleop_vr_leader --robot toddlerbot_2xm --ip 192.168.0.237
```
> Replace `192.168.0.237` with **toddy’s IP**.
