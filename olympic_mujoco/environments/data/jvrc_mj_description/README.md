# jvrc_mj_description

![Screenshot from 2022-12-27 18-09-23](https://user-images.githubusercontent.com/16384313/209642619-a35813fd-4cf1-4d80-bb92-b0c2f6d8b781.png)

JVRC1 model files for MuJoCo. The package will install the files in this directory so that [mc_mujoco](https://github.com/rohanpsingh/mc_mujoco) can pick them up automatically.

Install
-------

```bash
mkdir build
cd build
cmake ../
make && sudo make install
```

CMake options
-------------

- `SRC_MODE` if `ON` the files loaded by mujoco will point to the source rather than the installed files (default `OFF`)
