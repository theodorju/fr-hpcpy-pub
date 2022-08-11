# High Performance Computing with Python Project in Fluid Dynamics - Uni Freiburg

<img src="https://github.com/theodorju/fr-hpcpy-pub/blob/main/src/data/gifs/sliding_lid_velocity_field_0.1velocity_1.7omega_10000steps_300x300.gif" width="300" height="300" />

This repository is supplementary to the report and contains the python implementation of the experiments discussed there.

All implementations are under the `src` directory. Each implementation supports a number of command line arguments. Executing without any arguments will invoke the default behavior with the settings described in the report. 

The suggested way to execute the experiments is from within an anaconda environment with Python version 3.8 and the requirements specified in `requirements.txt` by executing the following:
```
conda create --name hpc python=3.8 -y
conda activate hpc
pip install -U pip
pip install -r requirements.txt
```

All experiments support `-h` as command line argument, which displays a helpful message describing their respective command line arguments.

---
All the example calls presented here assume that the user has 
1. Cloned the repository and `cd`-ed into the `/src` directory.
2. Created and activated an anaconda environment based on the provided `requirements.txt` file

Shear Wave Decay
---

Description of arguments:

```
optional arguments:
  -h, --help            show this help message and exit
  -d, --density         Execute sinusoidal density experiment.
  -v, --velocity        Execute sinusoidal velocity experiment.
  -f, --frequency       Execute multiple sinusoidal velocity experiments with varying 
                        collision frequency values.
                        
  -fv FREQUENCY_VALUES [FREQUENCY_VALUES ...], --frequency_values FREQUENCY_VALUES [FREQUENCY_VALUES ...] 
                        Space separated list of omega values. For example: -fv 1.1 1.2 1.4
  
  -g GRID_SIZE [GRID_SIZE ...], --grid_size GRID_SIZE [GRID_SIZE ...]
                        Space separated list of grid size (dim_0, dim_1). For example: -g 50 50
  
  -o OMEGA, --omega OMEGA   The collision frequency. Default value is 1.
  
  -s STEPS, --steps STEPS   The simulation steps. Default value is 2000.

```

Example call: Execute shear wave decay simulation with sinusoidal velocity initialization on a 50x50 grid with collision frequency 0.8 for 2000 simulation steps:  
```
python shear_wave_decay.py -v -o 0.8 -g 50 50 -s 2000
```

Couette Flow
---
Description of arguments:
```
optional arguments:
  -h, --help            show this help message and exit
  -a, --annotate        If provided the generated plot for the velocity profile 
                        will include the boundaries and an arrow indicating the direction 
                        of the velocity of the moving boundary. Default value is 
                        
  -v VELOCITY, --velocity VELOCITY
                        The velocity of the moving boundary. Currently only top moving 
                        boundary is supported.
                        
  -o OMEGA, --omega OMEGA
                        The collision frequency. Default value is 0.8.
                        
  -s STEPS, --steps STEPS
                        The simulation steps. Default value is 10000.
                        
  -g GRID_SIZE [GRID_SIZE ...], --grid_size GRID_SIZE [GRID_SIZE ...]
                        Space separated list of grid size (dim_0, dim_1). 
                        For example: -g 50 50
```

Example call: Execute Couette flow simulation with a velocity of 0.1, collision frequency equal to 0.8 on a 100x100 grid for 10000 steps. Annotate the velocity field graph with an arrow denoting the velocity and highlight the boundaries.
```
python couette_flow.py -a -v 0.1 -o 0.8 -s 10000 -g 100 100
```

Poiseuille Flow
---
Description of arguments:
```
  -h, --help            show this help message and exit
  -o OMEGA, --omega OMEGA
                        The collision frequency. Default value is 0.8.
                        
  -dss DENSITY_STEADY_STATE, --density_steady_state DENSITY_STEADY_STATE 
                        The density at each point on the grid at steady state. 
                        Current implementation assumes that all points of the grid 
                        have thesame density at the start of the experiment. 
                        Default = 1.
                        
  -din DENSITY_INPUT, --density_input DENSITY_INPUT
                        The percentage of increase in density value at the input. 
                        Default value is 1% increase of the steady state density.
                        
  -dout DENSITY_OUTPUT, --density_output DENSITY_OUTPUT
                        The percentage of increase in density value at the input. 
                        Default value is 1% decrease of the steady state density.
                        
  -s STEPS, --steps STEPS
                        The simulation steps. Default value is 10000.
                        
  -g GRID_SIZE [GRID_SIZE ...], --grid_size GRID_SIZE [GRID_SIZE ...]
                        Space separated list of grid size (dim_0, dim_1). 
                        For example: -g 50 50
```

Example call: Execute Poiseuille flow simulation for 10000 steps on a 50x50 grid with collision frequency equal to 0.8. Use default values for the rest of the arguments.
```
python poiseuille_flow.py -o 0.8 -dss 0.01 -s 10000 -g 50 50
```

Sliding Lid
---
Description of arguments:
```
  -h, --help            show this help message and exit
  -o OMEGA, --omega OMEGA
                        The collision frequency. Default value is 0.8.
                        
  -v VELOCITY, --velocity VELOCITY
                        The velocity of the moving lid.
                        
  -s STEPS, --steps STEPS
                        The simulation steps. Default value is 10000.
  -i, --gif             Generate gif. Keep snapshots of the velocity field 
                        to generate a gif of the flow (needs call to additional
                        script after simulation finishes)
                        
  -g GRID_SIZE [GRID_SIZE ...], --grid_size GRID_SIZE [GRID_SIZE ...]
                        Space separated list of grid size (dim_0, dim_1). 
                        For example: -g 50 50.
```
Example call: Execute the sliding lid simulation on a 300x300 grid with collision frequency equal to 1.7, lid velocity equal to 0.1  for 10000 steps.
```
python sliding_lid.py -o 1.7 -v 0.1 -s 10000 -g 300 300
```


Sliding Lid - Parallel Execution
---
Description of arguments:
```
  -h, --help            show this help message and exit
  -b, --benchmarking    Pass this argument if measuring execution time to avoid saving 
                        velocity arrays.
                        Default value False.
                        
  -o OMEGA, --omega OMEGA
                        The collision frequency. Default value is 1.7.
                        
  -v VELOCITY, --velocity VELOCITY
                        The velocity of the moving lid.
                        
  -g GRID_SIZE [GRID_SIZE ...], --grid_size GRID_SIZE [GRID_SIZE ...]
                        Space separated list of grid size (dim_0, dim_1). 
                        For example: -g 50 50
                        
  -d DISC [DISC ...], --discretization DISC [DISC ...]
                        Discretization of domain to parallel processes.
                        For example: -d 2 2
                        
  -s STEPS, --steps STEPS
                        The simulation steps. Default value is 10000
```

Example call: Execute the sliding lid simulation in parallel on a 300x300 grid with collision frequency equal to 1.7, lid velocity equal to 0.1  for 10000 steps. Use 4 cores and 2x2 discretization.
```
mpirun -n 4 python sliding_lid_parallel.py -g 300 300 -d 2 2 -o 1.7 -s 10000 -v 0.1
```
