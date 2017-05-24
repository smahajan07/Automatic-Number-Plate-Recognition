# Automatic-Number-Plate-Recognition

This repo is for experimental purpose only and is incomplete in a number of ways. Will be creating a repo for the same using deep learning. 

Following dependencies:
OpenCV 2.4.9.1
Python 2.7
Numpy
Tesseract-OCR and PyTesseract (For character recognition)

----------------------------------------------

Process:

1. Use OpenCV to find potential number plate region 
2. Collect all positives and negatives
3. Train classifier using Haar
4. Detect number plate
5. Extract characters 
6. Use Tesseract for OCR 

Usage:

Sample data is provided in 'data' folder. Change fine name and run file named 'run.sh' from terminal. 

TODO:
Better classifier for number plates. (More data and better machine required)
Self trained classifier for characters.
Probably test it on live system.
