The code and data can be found at 
https://drive.google.com/file/d/1K0F2RaWBEuPFlAZymfCA1HQjnkrkSjke/view?usp=sharing which is a link to a zip file containing the data and the code.

The code for the analysis was written in Python 3.7.2. However, it will also work with Python 2.7, the only difference being the way print statements are formatted.

Scikit-learn, a machine learning library for Python, was used to do the implemenatations of the algorithms.

Installation:
This project requires numpy, pandas, sklearn to be installed

Datasets:

Following datasets were used in this analysis:
1) MNIST - This dataset is stored in the folder ./MNISTDataset, containing two csv files mnist_train.csv and mnist_test.csv containing training and testing data repectively

2) Breast Cancer - This daaset is stored in the folder ./BreastCancerDataset, containing two csv files breastCancer_train.csv and breastCancer_test.csv containing the training and testing data respectively

Folder Structure:

All python files for running the analyses are located in the main directory along with the folders for the datasets.

File Structure:
The file structure is as follows: AlgorithmName_HyperParameterName_DatasetName.py for tuning and AlgorithmName_DatasetName_Test.py for testing. This file structure was used so that each algorithm for each hyperparameter could be tested separately, since some algorithms take a lot longer to run. Also, this makes it easir to edit the files.

Running the files:
To run the files, following commands will work:
python filename.py
python3 filename.py

The output of the algoritms is printed to the console and the relevant information is stored in a csv file with the same file naming structure as above but with a .csv instead of a .py

The graphs were drawn using excel