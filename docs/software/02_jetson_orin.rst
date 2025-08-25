
.. _jetson_orin:

Jetson Orin
===========

Jetson Orin is the on-board compute for ToddlerBot. We recommend Jetson Orin NX 16GB due to its cost effectiveness.


Flash the System
-----------------

#. Follow the instructions on `this page <https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/#flash-jetpack>`_ to flash the system.
   We've also provided some tips below to help you through the process.

   - For the Enter Force Recover Mode, you can refer to this photo if the GIF on the page is not clear.

   .. image:: ../_static/jetson_flash_pins.png

   - Click on JP6.1 tab and download the image corresponding to Jetson Orin NX 16GB. The download link may be slow—it took us about an hour to download the image.
   
   .. important::
      JetPack 6.1 or newer is strongly recommended, as it includes TensorRT 10.3+, which significantly improves engine build time and inference speed. We only support TensorRT 10+ for stereo depth estimation.

#. After flashing, unplug the powercable, the USB-C cable and the jumper wire. Replug the power cable, the HDMI cable, the keyboard and the mouse.
   The system should boot up now. The start screen should look like this:

   .. image:: ../_static/jetson_start_screen.jpg

#. Enter your username and password.

#. Set the APP Partition to the max size.

#. No need to install the Chromium browser.

#. Power off the Jetson Orin and install the WiFi card like the photo below.

   .. image:: ../_static/jetson_wifi.jpg

#. Power on the Jetson Orin and connect to the WiFi.

#. Press :kbd:`Ctrl` + :kbd:`Alt` + :kbd:`T` to open the terminal. Run ``ifconfig`` to check the IP address of the Jetson Orin.
   Write down the IP address and the hostname of the Jetson Orin, e.g., ``192.168.0.237`` and ``toddy@toddy-desktop``.

#. Select the power mode on the top right corner to **0:MAXN**.

#. Now feel free to unplug the HDMI cable, the keyboard and the mouse. We will access the Jetson Orin through SSH for the following steps.

.. note::
   The USB-C port on Jetson is only for flashing, which means transfering data
   through this port won't work.


Set up Real-Time (RT) Kernel
--------------------------------
#. SSH into the Jetson Orin.

   .. code:: bash

      ssh toddy@toddy-desktop.local

#. Follow the instructions on `this page <https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html#real-time-kernel-using-ota-update>`__ to install the real-time kernel.
   We found that it's OK to ignore the warning message below, but please open a GitHub issue if you encounter any problem:

   .. code:: bash

      Errors were encountered while processing:
      nvidia-l4t-rt-kernel
      nvidia-l4t-rt-kernel-oot-modules
      nvidia-l4t-rt-kernel-headers
      nvidia-l4t-display-rt-kernel
      E: Sub-process /usr/bin/dpkg returned an error code (1)

#. After rebooting, check the kernel version with ``uname -r``. It should be ``5.15.136-rt-tegra``. The numbers may vary, but ``-rt-tegra`` should be there.


Additional Packages
-------------------------
#. Go through the steps in :ref:`setup`. Note that ``wget`` is handy for downloading the Miniforge installer from the terminal.


#. Install ``torch`` and ``torchvision``.

   .. tabs::

      .. group-tab:: JetPack 6.1 and above

         Follow the information on `this page <https://developer.nvidia.com/embedded/downloads>`__ and search for ``PyTorch for Jetson``
         to install ``torch``. For reference, we downloaded `the wheel file for PyTorch v2.5.0 with JetPack 6.1 (L4T R36.4) + CUDA 12.6 <https://seeedstudio88-my.sharepoint.com/:u:/g/personal/youjiang_yu_seeedstudio88_onmicrosoft_com/EWCZOBNb9C9AoZe-mt23jLABZk942Lf0yopVGFJFTeL5DA?e=o7epES>`__.

      .. group-tab:: JetPack 6.0 and below

         Follow the information on `this page <https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048>`__
         to install ``torch`` and ``torchvision``. For reference, we downloaded `the wheel file for PyTorch v2.3.0 with JetPack 6.0 (L4T R36.2 / R36.3) + CUDA 12.2 <https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl>`__.
         
   We find that the ``--content-disposition`` option is useful for downloading the file with the correct name:

   .. code:: bash

      wget --content-disposition <link/to/the/wheel>

   Assuming the toddlerbot conda environment is activated, install the wheels with:

   .. code:: bash

      pip install <path/to/the/wheel>
      
   Please do **NOT** install ``numpy`` when installing ``torch`` as it will install ``numpy 2.x`` and cause conflicts.

   Last but not least, run the following command to verify that ``jax`` and ``torch`` versions are compatible:

   .. code:: bash

      python examples/test_jax_torch.py --platform jetson

   If you encounter any issue with ``cuSPARSELt``, refer to `this page <https://developer.nvidia.com/cusparselt-downloads?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_network>`__ to download the correct version.

#. Install the ``ch9344ser`` driver for the 8-channel communication board:

   .. code:: bash

      git clone git@github.com:WCHSoftGroup/ch9344ser_linux.git
      cd ch9344ser_linux/driver/
      make
      sudo make install

#. We need to install Jetson.GPIO. Revised from instructions on `this page <https://github.com/NVIDIA/jetson-gpio>`__, we need to run the following commands to set it up:

   .. code:: bash

      sudo groupadd -f -r gpio
      sudo usermod -aG gpio $USER
      sudo chown root.gpio /dev/gpiochip0
      sudo chmod 660 /dev/gpiochip0

      conda activate toddlerbot
      sudo cp ~/miniforge3/envs/toddlerbot/lib/python3.10/site-packages/Jetson/GPIO/99-gpio.rules /etc/udev/rules.d/
      sudo udevadm control --reload-rules && sudo udevadm trigger

#. To access the imu and dynamixel motors, we need to add the user to i2c and dialout group.

   .. code:: bash

      sudo usermod -aG i2c $USER
      sudo usermod -aG dialout $USER


#. Now reboot the Jetson Orin. Run ``groups`` to check if the user is in the i2c, dialout, and gpio group. An example output looks like this:

   .. code:: bash

      toddy adm dialout cdrom sudo audio dip video plugdev render i2c lpadmin sambashare gdm weston-launch gpio


#. For the accuracy of teleoperation and logging over network, we need to
   install ntp package to sync time of the Jetson to server.

   .. code:: bash

      sudo apt install ntp ntpdate
      sudo systemctl enable ntp

      sudo nano /etc/ntp.conf

      comment out the following lines:

      # pool 0.ubuntu.pool.ntp.org iburst
      # pool 1.ubuntu.pool.ntp.org iburst
      # pool 2.ubuntu.pool.ntp.org iburst
      # pool 3.ubuntu.pool.ntp.org iburst
      # pool ntp.ubuntu.com

      add:

      server <ip address of the steam deck> iburst

      sudo systemctl start ntp

#. For the fisheye cameras to work, we need to install the following packages:

   .. code:: bash

      sudo apt install v4l-utils ffmpeg

#. For the speaker and microphone to work, we need to install the following packages:

   .. code:: bash

      sudo apt install portaudio19-dev flac

#. For stereo depth estimation using FoundationStereo + TensorRT:

   **Prerequisites:** Ensure JetPack 6.1 or above is installed.

   **Step 1:** Check your CUDA version and set up PyCUDA for GPU acceleration.

   Check CUDA version:

   .. code:: bash

      cat /usr/local/cuda/version.json

   Or check via ``jtop`` info tab. Then set up PyCUDA (replace ``12.6`` with your CUDA version):

   .. code:: bash

      export PATH=/usr/local/cuda-12.6/bin:$PATH
      export CUDA_ROOT=/usr/local/cuda
      pip install pycuda

   **Step 2:** Configure TensorRT for the conda environment.

   Add TensorRT to Python path:

   .. code:: bash

      echo "/usr/lib/python3.10/dist-packages" > $CONDA_PREFIX/lib/python3.10/site-packages/tensorrt_global.pth
