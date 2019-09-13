./bin/Pink --train ../scripts/data.bin som1.bin
./bin/Pink --layout hexagonal --som-width 7 --som-height 7 --euclidean-distance-type uint8 --train ../scripts/data.bin som2.bin
./bin/Pink --layout Cartesian --som-width 5 --som-height 7 --train ../scripts/data.bin som3.bin
./bin/Pink --layout Cartesian --som-width 5 --som-height 4 --som-depth 3 --train ../scripts/data.bin som4.bin
./bin/Pink --map ../scripts/data.bin map1.bin som1.bin