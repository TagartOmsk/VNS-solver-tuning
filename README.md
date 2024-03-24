# VNS-solver-tuning
First iteration of solver parameter tuner, based off [VNS algorithm](https://en.wikipedia.org/wiki/Variable_neighborhood_search). Currently works with Gurobi solver through GAMS interface.


## How to run
Current version adjusted to work with [multicore processor sheduling problem](https://github.com/mysosnovskaya/multicore-processor-scheduling-test-data).
It currently doesn't support config files or CLI parameters, so everything must be done through modifying VNS.py file.

Namely:
- prepare instances of the problem in GAMS format, put folder path into `inst_dir_path` variable;
- prepare sets file, put it's path into `sets` variable;
- make sure `discovered_points_file_path` does not exist or carefully saved on your file system or it'd be rewritten.
- make sure `requirements.txt` are satisfied

Further run syntax is `python VNS.py`.