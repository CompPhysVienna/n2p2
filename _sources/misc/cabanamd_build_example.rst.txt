.. _cabanamd_build_example:

CabanaMD GPU build example
==========================

.. warning::

   This is not a fully tested installation guide but rather a collection of
   build instructions that worked for a specific system at a certain point in
   time. They are kept here only as a reference, for your own build please refer
   to the current build instructions of the individual software packages.

System specifications
---------------------

-  CPU: ``Intel(R) Core(TM) i7-7800X (6/12 cores/threads)``
-  RAM: ``32GB``
-  OS: ``Linux Mint 19.3``
-  GPU: ``GeForce GTX 1060 6GB (GP106, Pascal architecture)``
-  CUDA version: ``11.0``
-  CUDA capability: ``6.1``
-  GPU driver: ``450.51.06``

Build steps
-----------

In order to have all packages in one place please first create a working directory for
these instructions (e.g. ``CabanaMD-Setup``) and ``cd`` there.

On this particular system it was required to build a CUDA-aware MPI. Create a
file ``build_openmpi.sh``, copy these commands and execute it (or go through the
commands one by one):

.. code-block:: bash

   #!/bin/bash

   wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.0.tar.gz
   tar -xvzf openmpi-4.1.0.tar.gz
   mv openmpi-4.1.0 openmpi
   
   cd openmpi
   mkdir build
   ./configure --prefix=`pwd`/build --with-cuda=/usr/local/cuda-11.0 && \
   make all -j && \
   make install
   cd ..

After successful compilation you should see binaries in ``openmpi/build/bin/``.
Be sure to set the path to your CUDA installation correctly in the
``./configure`` line.

Next, we download `Kokkos <https://github.com/kokkos/kokkos>`__ and compile it.
Create and execute a file ``build_kokkos.sh`` with this content:

.. code-block:: bash

   #!/bin/bash

   git clone git@github.com:kokkos/kokkos.git
   export COMPILER=`pwd`/kokkos/bin/nvcc_wrapper
   export ARCH=sm_61
   # CAUTION: Also set the flag (-D Kokkos_ARCH_PASCAL61=On)
   cd kokkos
   git checkout 3.2.00
   mkdir build
   cd build
   cmake ../ \
    -D CMAKE_CXX_COMPILER=${COMPILER} \
    -D CMAKE_CXX_FLAGS=-arch=${ARCH} \
    -D CMAKE_INSTALL_PREFIX=./install \
    -D Kokkos_CUDA_DIR=/usr/local/cuda-11.0/ \
    -D Kokkos_ENABLE_SERIAL=On \
    -D Kokkos_ENABLE_OPENMP=On \
    -D Kokkos_ENABLE_CUDA=On \
    -D Kokkos_ENABLE_CUDA_LAMBDA=On \
    -D Kokkos_ENABLE_CUDA_UVM=On \
    -D Kokkos_ARCH_PASCAL61=On \
    -D Kokkos_ENABLE_HWLOC=On &&
   make install -j
   cd ../..

Make sure to set the ``ARCH`` and ``Kokkos_ARCH_???`` correctly for your GPU
(see `docs <https://github.com/kokkos/kokkos/blob/master/BUILD.md>`__). This
should create a directory ``kokkos/build/install`` where you can find libraries
and headers in the ``lib`` and ``include`` subfolder, respectively.

Now we install `Cabana <https://github.com/ECP-copa/Cabana>`__, create
``build_Cabana.sh``, fill in below commands and execute it:

.. code-block:: bash

   #!/bin/bash

   git clone git@github.com:ECP-copa/Cabana.git
   cd Cabana
   # No specific version is specified in the docs, only "master". If that does not
   # work at a later point in time use this line to get a commit that worked:
   git checkout 5c2503aa72c3e3cb4db34ea52e41a2ac446e5719
   export ARCH=sm_61
   export OPENMPI_BUILD_DIR=`pwd`/../openmpi/build
   export KOKKOS_SRC_DIR=`pwd`/../kokkos
   export KOKKOS_INSTALL_DIR=${KOKKOS_SRC_DIR}/build/install
   mkdir build
   cd build
   cmake ../ \
    -D CMAKE_BUILD_TYPE="Release" \
    -D CMAKE_PREFIX_PATH="${OPENMPI_BUILD_DIR};${KOKKOS_INSTALL_DIR}" \
    -D CMAKE_INSTALL_PREFIX=`pwd`/install \
    -D CMAKE_CXX_COMPILER=${KOKKOS_SRC_DIR}/bin/nvcc_wrapper \
    -D CMAKE_CXX_FLAGS=-arch=${ARCH} \
    -D MPI_CXX_COMPILER=${OPENMPI_BUILD_DIR}/bin/mpic++ \
    -D Cabana_REQUIRE_CUDA=On \
    -D Cabana_ENABLE_MPI=On \
    -D Cabana_ENABLE_TESTING=OFF \
    -D Cabana_ENABLE_EXAMPLES=OFF && \
   make install -j
   cd ../..

Again, check that the ``ARCH`` variable is set correctly. Since we have excluded
the tests and examples (``Cabana_ENABLE_TESTING=OFF`` and
``Cabana_ENABLE_EXAMPLES=OFF``) this will not compile anything but create some
headers in ``build/install/include``.

The last step before we get to CabanaMD is to set up n2p2, create
``build_n2p2.sh`` with the follwing lines and execute:

.. code-block:: bash

   #!/bin/bash

   git clone https://github.com/CompPhysVienna/n2p2
   cd n2p2
   git checkout v2.0.1
   cd src
   make libnnpif -j
   cd ../..

Now there should be headers in ``n2p2/include``.

Finally, compile `CabanaMD <https://github.com/ECP-copa/CabanaMD>`__, create a
file ``build_CabanaMD.sh``, fill it with this content and execute (check again
the ``ARCH`` variable):

.. code-block:: bash

   #!/bin/bash

   git clone git@github.com:ECP-copa/CabanaMD.git
   cd CabanaMD
   # No specific version is specified in the docs, only "master". If that does not
   # work at a later point in time use this line to get a commit that worked:
   git checkout 15fa51c03a388c6c7f59dbbd4e21ecc3e5c7459b
   export ARCH=sm_61
   export OPENMPI_BUILD_DIR=`pwd`/../openmpi/build
   export KOKKOS_SRC_DIR=`pwd`/../kokkos
   export CABANA_DIR=`pwd`/../Cabana/build/install
   export N2P2_DIR=`pwd`/../n2p2
   mkdir build
   cd build
   cmake ../ \
    -D CMAKE_BUILD_TYPE="Release" \
    -D CMAKE_PREFIX_PATH="${OPENMPI_BUILD_DIR};${CABANA_DIR}" \
    -D CMAKE_INSTALL_PREFIX=`pwd`/install \
    -D CMAKE_CXX_COMPILER=${KOKKOS_SRC_DIR}/bin/nvcc_wrapper \
    -D CMAKE_CXX_FLAGS=-arch=${ARCH} \
    -D MPI_CXX_COMPILER=${OPENMPI_BUILD_DIR}/bin/mpic++ \
    -D Cabana_ENABLE_MPI=On \
    -D N2P2_DIR=${N2P2_DIR} \
    -D CabanaMD_VECTORLENGTH=32 \
    -D CabanaMD_ENABLE_NNP=On \
    -D CabanaMD_MAXSYMMFUNC_NNP=30 \
    -D CabanaMD_VECTORLENGTH_NNP=1 \
    -D CabanaMD_ENABLE_TESTING=OFF && \
   make -j
   cd ../..

If successful you will now find the CabanaMD executable ``cbnMD`` in
``CabanaMD/build/bin/``.

Running an example on the GPU
-----------------------------
Finally, we can test it on the GPU with the provided NNP example. Create
``test_CabanaMD_GPU_NNP.sh`` with below content and execute:

.. code-block:: bash
   
   #!/bin/bash

   cd CabanaMD
   build/bin/cbnMD -il input/in.nnp --device-type CUDA
   mv cabanaMD.out ..
   mv cabanaMD.err ..
   cd ..

You should see the (slightly modified) n2p2 library output and in the output
file ``cabanaMD.out`` you will find some information about your GPU and the
performed timesteps as in:

.. code-block:: none

   Read input file.
   macro  KOKKOS_ENABLE_CUDA      : defined
   macro  CUDA_VERSION          = 11000 = version 11.0
   Kokkos::Cuda[ 0 ] GeForce GTX 1060 6GB capability 6.1, Total Global Memory: 5.927 G, Shared Memory per Block: 48 K : Selected
   Using: SystemVectorLength: 32 System:1AoSoA
   Using: SystemNNPVectorLength: 1 NNPSystem:1AoSoA
   Using: Force:NNPCabana Neighbor:CabanaVerletFull Comm:CabanaMPI Binning:CabanaLinkedCell Integrator:NVE
   Atoms: 108 108
   Created atoms.
   
   #Timestep Temperature PotE ETot Time Atomsteps/s
   0 600.000000 -5.780009 -5.703171 0.00 0.00e+00
   10 577.114210 -5.777077 -5.703170 0.50 2.14e+03
   20 512.839729 -5.768845 -5.703169 0.97 2.32e+03
   30 418.982352 -5.756824 -5.703168 1.43 2.35e+03
   40 312.732989 -5.743216 -5.703167 1.90 2.31e+03
   50 213.508287 -5.730508 -5.703165 2.36 2.33e+03
   60 138.634903 -5.720919 -5.703165 2.83 2.30e+03
   70 99.064735 -5.715851 -5.703164 3.29 2.34e+03
   80 96.861148 -5.715569 -5.703165 3.76 2.28e+03
   90 125.599583 -5.719251 -5.703166 4.23 2.32e+03
   100 173.352777 -5.725367 -5.703167 4.70 2.29e+03
   
   #Procs Atoms | Time T_Force T_Neigh T_Comm T_Int T_Other |
   1 108 | 4.70 4.31 0.00 0.38 0.00 0.00 | PERFORMANCE
   1 108 | 1.00 0.92 0.00 0.08 0.00 0.00 | FRACTION
   
   #Steps/s Atomsteps/s Atomsteps/(proc*s)
   2.13e+01 2.30e+03 2.30e+03

You can check your GPU utilization with the ``nvidia-smi`` tool, run ``watch
nvidia-smi`` in another shell. The executable ``cbnMD`` should be
listed under ``Processes``:

.. code-block:: none

   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
   |-------------------------------+----------------------+----------------------+
   | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   |                               |                      |               MIG M. |
   |===============================+======================+======================|
   |   0  GeForce GTX 106...  On   | 00000000:65:00.0 Off |                  N/A |
   |  0%   49C    P2    33W / 200W |    476MiB /  6069MiB |     96%      Default |
   |                               |                      |                  N/A |
   +-------------------------------+----------------------+----------------------+
   
   +-----------------------------------------------------------------------------+
   | Processes:                                                                  |
   |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
   |        ID   ID                                                   Usage      |
   |=============================================================================|
   |    0   N/A  N/A      1385      G   /usr/lib/xorg/Xorg                224MiB |
   |    0   N/A  N/A      2402      G   cinnamon                           48MiB |
   |    0   N/A  N/A     10163      C   build/bin/cbnMD                    71MiB |
   +-----------------------------------------------------------------------------+
