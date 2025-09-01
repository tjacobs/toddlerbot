#!/bin/bash

python examples/test_teleoperation_gmr.py

# test teleoperation + simulator

# in one terminal run:
python toddlerbot/policies/run_policy.py --policy teleop_follower_pd --robot toddlerbot_2xc --vis view --task teleop_vr
# in the second terminal run:
python toddlerbot/policies/run_policy.py --policy teleop_vr_leader --robot toddlerbot_2xc

# test teleoperation + real robot
# on toddy run:
python toddlerbot/policies/run_policy.py --policy teleop_follower_pd --robot toddlerbot_2xm --sim real --task teleop_vr --ip 192.168.0.203
# replace the ip with your desktop ip
# on your desktop run:
python toddlerbot/policies/run_policy.py --policy teleop_vr_leader --robot toddlerbot_2xm --ip 192.168.0.237
# replace the ip with toddy's ip