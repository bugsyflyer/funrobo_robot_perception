# funrobo_robot_perception (Module 5)

This repository accompanies the class activities in module 5 with a focus on helping you learn the following:
1. Basics of digital image representation
2. Basic digital image manipulation and processing
3. 2D object detection and segmentation using traditional CV methods and deep learning methods
4. 3D object pose estimation
5. General vision-based robot scene understanding and representation

These activities will be performed using Jupyter Notebooks.

## Setting up your PC

- You can complete this assignment on any computer OS: Mac, Windows, Linux, etc. All you need is Python 3 interpreter (code was tested using Python 3.11).

### Step 0 (Optional): Install VScode
- Install Visual Studio Code (I strongly recommend using this, if you don’t already do. It’s the best IDE in my humble opinion)
- Follow the instructions [here to install.](https://code.visualstudio.com/download)



### Step 1: Install Python 3 (if not already installed)
- First, check if you have Python3, to do that, open your terminal and type:
```bash
python3 --version     # <--- type this
Python 3.10.12          # <--- you should see something like this
```
- If you don’t have Python installed, follow this [tutorial here](https://realpython.com/installing-python/) to install it.


### Step 2: Get this repository from Github
I recommend that one teammate forks the repository, and others clone from that fork.
- Fork this repository on Github
- Clone your fork:
```base
$ git clone <YOUR_FORK_URL>
$ cd funrobo_robot_perception
```


### Step 3: Create a conda environment
- From the root of the repository (same folder as ``environment.yml``)
```bash
# cd to the project folder
cd funrobo_robot_perception
conda env create -f environment.yml
```
- Activate the environment:
```bash
conda activate funrobo_vision
```

### Step 4: Manually install these packages using PIP
- Run the following commands:
```bash
pip install roboflow supervision machinevision-toolbox-python spatialmath-python
```

### Step 4: Register the jupyter notebook kernel
- Activate the environment:
```bash
python -m ipykernel install --user --name funrobo_vision --display-name "Funrobo Vision Env"
```


### How to Run

- If setup worked well, you should be able to open the following Jupyter notebooks in VS Code:
    - [**class_activity_1.ipynb**](https://github.com/OlinCollege-FunRobo/funrobo_robot_perception/blob/main/class_activity_1.ipynb)
    - [**class_activity_2.ipynb**](https://github.com/OlinCollege-FunRobo/funrobo_robot_perception/blob/main/class_activity_2.ipynb)
    - [**class_activity_3.ipynb**](https://github.com/OlinCollege-FunRobo/funrobo_robot_perception/blob/main/class_activity_3.ipynb)


**NOTE:** Please make sure you select the python interpreter associated with the virtual environment you just created when running the Jupyter notebook.