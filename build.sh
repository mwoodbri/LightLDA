# build lightlda

git clone -b multiverso-initial https://github.com/hhaoyan/Multiverso.git

sed -i 's/-lmpich -lmpl //' Makefile
sed -i 's/mpic++/mpicxx/' Makefile Multiverso/Makefile
sed -i 's/-lmpl //' Multiverso/Makefile
sed -i 's/mpich mpl //' src/multiverso/CMakeLists.txt src/multiverso_server/CMakeLists.txt

cd Multiverso
cd third_party
sh install.sh
cd ..
make -j4 all

cd ..
make -j4
