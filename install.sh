rm -rf python/dist/*
make -j4 || exit 1
pip3 uninstall paddlepaddle -y
pip3 install -U python/dist/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl
