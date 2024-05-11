# Linear Regression

Linear Regression with one variable to predict profits for a restaurant franchise.

# Outline

- 1 - Packages
- 2 - Linear regression with one variable
  - 2.1 Problem Statement
  - 2.2 Dataset
  - 2.3 Refresher on linear regression
  - 2.4 Compute Cost
    - Computing Cost
  - 2.5 Gradient descent
    - Implementing Gradient Descent
  - 2.6 Learning parameters using batch gradient descent

## Packages

- [numpy](www.numpy.org) is the fundamental package for working with matrices in Python.
- [matplotlib](http://matplotlib.org) is a famous library for plotting graphs in Python.
- `utils.py` contains helper functions for this assignment.

## Problem Statement

Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.

- You would like to expand your business to cities that may give your restaurant higher profits.
- The chain already has restaurants in various cities and you have data for profits and populations from the cities.
- You also have data on cities that are candidates for a new restaurant.
  - For these cities, you have the city population.

Can you use the data to help you identify which cities may potentially give your business higher profits?

## Dataset

You will start by loading the dataset for this task.

- The `load_data()` function shown below loads the data into variables `x_train` and `y_train`
  - `x_train` is the population of a city
  - `y_train` is the profit of a restaurant in that city. A negative value for profit indicates a loss.
  - Both `X_train` and `y_train` are numpy arrays.

## About variables

`x_train` is a numpy array that contains decimal values that are all greater than zero.

- These values represent the city population times 10,000
- For example, 6.1101 means that the population for that city is 61,101.

Similarly, `y_train` is a numpy array that has decimal values, some negative, some positive.

- These represent your restaurant's average monthly profits in each city, in units of \$10,000.
  - For example, 17.592 represents \$175,920 in average monthly profits for that city.
  - -2.6807 represents -\$26,807 in average monthly loss for that city.

## Acknowledgments:

This demonstration of linear regression from scratch is part of my learning journey of understanding and implementing ml algos from scratch. To get the in-depth understanding of working of this code or linear regression in general, please refer to https://www.youtube.com/playlist?list=PLb0Gp98iu3OyY9zWJfSMq26nmkNKztNhA
