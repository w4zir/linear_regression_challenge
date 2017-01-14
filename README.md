# linear_regression_challenge
This is the code for challenge in "How to Make a Prediction - Intro to Deep Learning #1' by Siraj Raval on YouTube

##Overview
This is the code for challenge at the end of [this](https://youtu.be/vOppzHpvTiQ) video by Siraj Raval on Youtube. The goal is to predict an animal's body weight given it's brain weight on a dataset provided. I have [Linear Regression](http://www.statisticssolutions.com/what-is-linear-regression/) on the dataset provided as part of the challenge that include a list of brain weight and body weight measurements from a bunch of animals. First I fit a line to the data using the scikit learn machine learning library, then I sample a random point (brain, body) from the data and predict body size from brain size using our trained model. I print the error between the predicted body size and the actual body size. Lastly I plot our graph using matplotlib along with the predicted point (in red color) and actual point (in green).

![](/plots/linear_regression_challenge.jpg?raw=true "Acutal (gree) and Predicted (red) body size.")

##Dependencies

* pandas
* scikit-learn
* matplotlib
* random

You can just run
`pip install -r requirements.txt` 
in terminal to install the necessary dependencies. Here is a link to [pip](https://pip.pypa.io/en/stable/installing/) if you don't already have it.

##Usage

Type `python linear_regression.py` into terminal and you'll see the sample point and error on console. The data is also plotted on using scatter plot and line of best fit appear is also shown.


##Credits

The code is an extension of the original code by [Siraj](https://github.com/llSourcell).

