# Martian Bow Shock Detection
Automatic Detection of Martian bow shock crossings using data of the Mars Express mission: A Deep Learning Approach to see more chek this link : 
https://pnst-2022.sciencesconf.org/399962


<h2>Description</h2>
<p>This repository contains a python script (run_predictions_orchestra.py) which should be executed in 
a virtual environment in order to make prediction of Martian Bow Shock crossings associated with a given start and stop times.
<h2>How this work ?</h2>
To do predictions you need to follow these steps:
👏
<ul>
    <li>activate the virtual environment = venv\Scripts\activate</li>
    <li>install the requirements = pip --no-cache-dir install -r path/../requirements.txt</li>
    <li>run the script with the following arguments a destination folder <strong>path</strong> to store results, a start time <strong>start</strong> and 
    a stop time <strong>stop</strong> =>
    <strong> python -m run_predictions_orchestra path start stop</strong>
    </li>
</ul>
<p>For example: </p>
python -m run_predictions_orchestra . 2008-07-03 2008-07-05


 👏
