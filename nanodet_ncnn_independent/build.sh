if [ ! -d "./build" ]
then
	echo "Creating directory ./build"
	mkdir ./build
fi

cd ./build
#cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug
cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make -j6
cd ..
