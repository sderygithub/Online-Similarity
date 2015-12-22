"""
Generic Gaussian Functionalities
Used by the prototype Bayesian Network

Provides Gaussian Density functions and approximation to Gaussian CDF.
(https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution)
(https://en.wikipedia.org/wiki/Normal_distribution#Numerical_approximations_for_the_normal_CDF)

Usage:

Options:

Examples:

License:

Copyright (c) 2015 Sebastien Dery

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import math
from collections import defaultdict
from itertools import combinations, product



b0 = 0.2316419
b1 = 0.319381530
b2 = -0.356563782
b3 = 1.781477937
b4 = -1.821255978
b5 = 1.330274429


def std_gaussian_cdf(x):
    """
    https://en.wikipedia.org/wiki/Normal_distribution#Numerical_approximations_for_the_normal_CDF
    """
    g = make_gaussian(0, 1)
    t = 1 / (1 + b0 * x)
    return 1 - g(x) * (
        b1 * t +
        b2 * t ** 2 +
        b3 * t ** 3 +
        b4 * t ** 4 +
        b5 * t ** 5)


def make_gaussian(mean, std_dev):

    def gaussian(x):
        return 1 / (std_dev * (2 * math.pi) ** 0.5) * \
            math.exp((-(x - mean) ** 2) / (2 * std_dev ** 2))

    gaussian.mean = mean
    gaussian.std_dev = std_dev
    gaussian.cdf = make_gaussian_cdf(mean, std_dev)

    return gaussian


def make_gaussian_cdf(mean, std_dev):

    def gaussian_cdf(x):
        t = (x - mean) / std_dev
        if t > 0:
            return std_gaussian_cdf(t)
        elif t == 0:
            return 0.5
        else:
            return 1 - std_gaussian_cdf(abs(t))

    return gaussian_cdf


def make_log_normal(mean, std_dev, base=math.e):

    def log_normal(x):

        return 1 / (x * (2 * math.pi * std_dev * std_dev) ** 0.5) * \
            base ** (-((math.log(x, base) - mean) ** 2) / (2 * std_dev ** 2))

    log_normal.cdf = make_log_normal_cdf(mean, std_dev)

    return log_normal


def make_log_normal_cdf(mean, std_dev, base=math.e):

    def log_normal_cdf(x):
        gaussian_cdf = make_gaussian_cdf(0, 1)
        return gaussian_cdf((math.log(x, base) - mean) / std_dev)

    return log_normal_cdf


