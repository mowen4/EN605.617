cd npp/
make
./nppassignment -input=baboon.ascii.pgm
./nppassignment -input=foliage.ascii.pgm
cd ..
cd nvgraphAssignment/
make
./nvgraphAssignment -node=0
./nvgraphAssignment -node=2
cd ..
cd thrustAssignment/
make
./thrust
./thrust 10
./thrust 100
