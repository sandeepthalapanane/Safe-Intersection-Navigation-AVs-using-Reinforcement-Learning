# Installation

1) SUMO

    Install SUMO using this link: https://sumo.dlr.de/docs/Installing/index.html\

    For easy installation of dependencies and running the reinforcement
    learning model, we recommend installing SUMO in a Windows environment using the conda environment.

2) Anaconda

    Install Anaconda using this link: https://docs.anaconda.com/free/anaconda/install/windows/


After Successful installation of both SUMO and Anaconda/Miniconda, download the folder and create a conda environment using the `enviornment.yaml` file in the folder using the command given below:

````
cd <folder_location\\folder_name>
conda env create -f environment.yaml
````

* Changes to be made in Sumo.py file
    * cd gym_sumo\\\envs
    * locate and open Sumo.py
    * Change sumoBinary location inside the `initSimulator` function in both if and else statements (# comments will be provided in the code to make sure you locate the code line)

After creating the conda environment with the required dependencies using the file, to activate the environment use the command given below:

````
conda activate sumo_av_gym
````


## Training Phase

To train the autonomous vehicle using the provided environments, run the following command on your terminal

- Make sure to be inside the base folder which should be `Sumo_AV_Gym`

````
python train.py
````

If you want to see the sumo environment gui while training, change the `self.withGUI = False` to `self.withGUI = True` inside the `SUMOEnv class` -> `_init_ function` -> `mode="train"` in the `sumo_env.py` file

It is suggeested to run the training without the GUI to speed up the training process


## Testing Phase

To test the pretrained rl model (which are already provided within the folder inside the models folder), run the following command on your terminal

 - Make sure to be inside the base folder which should be `Sumo_AV_Gym`

````
python test.py --model 1200_episode_model.pth
````

You are free to choose any model you want to test which is in the models folder but recommended to test using the `1200_episode_model.pth`