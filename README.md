# Martian Bow Shock Detection
Automatic Detection of Martian bow shock crossings using data of the Mars Express mission: A Deep Learning Approach to see more chek this link : 
https://pnst-2022.sciencesconf.org/399962.
# Abstract
*We investigate to automatically detect the Martian bow shock crossings using the data of the Mars Express mission provided by CDPP-AMDA. 
Using a Multilayer Perceptron Neural Network, we provide an automatic classifier to predict the
Martian bow shock crossings. A published catalog with around 11800 bow shock
crossings has been used for labeling the data [1]. The challenging task was to deal with
the unbalanced data, indeed, in our dataset, we have unequal distribution of classes:
shocks and no shocks. Classification of unbalanced data is a difficult task because
there are so few samples (shocks) to learn from. To tackle this problem is to penalize
the mis-classification made by the minority class by setting a higher class weight and at
the same time reducing weight for the majority class.* </br> </br>
<sup>[1] B. E. S. Hall et al. “Annual variations in the Martian bow shock location as observed by the Mars Express mission”. In: Journal of
Geophysical Research: Space Physics 121.11 (2016), pp. 11, 474–11, 494</sup>

<h2>Description</h2>
<p>This repository contains a python script (run_predictions_orchestra.py) which should be executed in 
a virtual environment in order to make prediction of Martian Bow Shock crossings associated with a given start and stop times.
<h2>How this work ?</h2>
To do predictions you need to follow these steps:
👏
<ul>
    <li>you should have python >= 3.10  already installed
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
