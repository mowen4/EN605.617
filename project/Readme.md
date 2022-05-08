Use make to compile the program. 
./project.exe to run it
./project 5 to change the code to only run 5 orders of magnitude 10
./project 9 will be for 100M records.
The program will by default run 10M records, vocareum did not support 100M of memory objects

requires c++11 and latest cuda dev kit

The python code can be run with the latest standard conda install of python using "python project.py"
it will run as a default for 10M records as well. that can be changed by altering line 28

The python code serves as a reference point for the incremental GPU acceleration
