# build lightlda

git clone -b multiverso-initial https://github.com/hhaoyan/Multiverso.git

cd Multiverso
cd third_party
sh install.sh
cd ..
make -j4 all

cd ..
make -j4
