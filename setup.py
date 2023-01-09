#!/usr/bin/env python

from setuptools import setup

setup(
    name="SLAM_diffusion",
    py_modules=["SLAM_diffusion"],
    install_requires=["torch", "tqdm"],
    author="Alexy Skoutnev",
    author_email='alexyskoutnev@gmail.com',
    version="0.0.0",
    description='SLAM application utilizing a diffusion model',
    url="https://github.com/Alexyskoutnev/SLAM_Diffusion"
)
