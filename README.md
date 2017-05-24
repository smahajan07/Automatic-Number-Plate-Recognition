# Automatic-Number-Plate-Recognition

This repo is for experimental purpose only and is incomplete in a number of ways. Will be creating a repo for the same using deep learning. 

Following dependencies:
1. OpenCV 2.4.9.1
2. Python 2.7
3. Numpy
4. Tesseract-OCR and PyTesseract (For character recognition)

----------------------------------------------

Process:

1. Use OpenCV to find potential number plate region 
2. Collect all positives and negatives
3. Train classifier using Haar
4. Detect number plate
5. Extract characters 
6. Use Tesseract for OCR 

Usage:

Make a folder 'testing' and 'testing/Numbers'. Every time you run, output will be saved here (Also, delete output if running code again). Sample data is provided in 'data' folder. Change file name in code and run file named 'run.sh' from terminal. 

TODO:
1. Better classifier for number plates. (More data and better machine required)
2. Self trained classifier for characters.
3. Probably test it on live system.
