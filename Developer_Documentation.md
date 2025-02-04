
# Aegean

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Testing](#testing)

## Introduction
Aegean is a source finding program designed to find and characterise compact continuum radio sources. 

## Installation
To install Aegean, follow the steps below.
### Prerequisites
-	Python
-	OpenMPI
-  Git

### Steps
1. Clone the repository.
   ```
   git clone https://github.com/PaulHancock/Aegean.git
   ```
2. Navigate to the project directory.
   ```
   cd AEGEAN
   ```
3. Install.
   ```
   pip install .
   ```

## Usage
Aegean can be used as a command line tool. The command line tool can be used to find sources in a FITS image and output the results in a variety of formats. The command line tool can also be used to load background and rms files that might have been previously created by BANE.
### Basic Usage
```
aegean path/to/file
```

### Advanced Usage

There are many options available for the Aegean command line tool. These include:
-- table : for specifying the format of the output
-- autoload : for automatically loading the background and rms files that might have been previously created by BANE

### Example
```
aegean path/to/file --table table_name.csv,table_name.fits --autoload
```

### MPI usage
To use the you can use the following bash script with minor modifications to suit your needs.
```
#!/bin/bash -l
#SBATCH --account=pawsey000X
#SBATCH --partition=work
#SBATCH --time=00:40:00
#SBATCH --nodes=16
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32gb
#SBATCH --output=/scratch/pawsey000X/user/<directory>/<file_name>.o%A
#SBATCH --error=/scratch/pawsey000X/user/<directory>/<file_name>.e%A

cd /scratch/pawsey000X/<user>/<directory>
source ../env/bin/activate
module load cray-mpich/8.1.25
module list
echo $HOSTNAME
srun aegean --table ${SLURM_JOB_ID}.fits --autoload <file_name>
```

## Testing
To run tests for the new addSuffix function, run the following file:
```
pytest tests/unit/test_addSuffix.py 
```

## References
For further user guide and documentation, please refer to the Aegean Github repository at https://github.com/PaulHancock/Aegean
