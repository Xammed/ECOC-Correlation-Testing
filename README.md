# ECOC-Correlation-Testing
Programs for testing correlation between classifiers in ensemble learning via error correcting output codes.

MasterTester Is most up to date, use on the commandline as follows:

python3 MasterTester.py "datasets/pendigits.csv" -1 0 12 10 247 1 0 10 1 0.2 "./penDigits_cmd.txt"

Order of arguments: labelscolumn, databegin, dataend, numclasses, stringofmodels, graphing, printing, folds, starting_p, ending_p, file_for_output

Where starting p is the top row of the results at the end, and ending p is the bottom row.

Example with NO variation: python3 MasterTester.py "datasets/pendigits.csv" -1 0 12 10 247 1 0 10 1 1 "./penDigits_cmd.txt"
(starting_p = 1, ending_p = 1)

Example for data with 0.9 down to 0.5: Example with NO variation: python3 MasterTester.py "datasets/pendigits.csv" -1 0 12 10 247 1 0 10 .9 .5 "./penDigits_cmd.txt"



Or, run by yourself by writing code in the else branch at the bottom of the file (if you want to do anything extra). If you don't specify anything on the commandline, the code written in the else branch will exectue. That's blank right now but available for users to enforce functionality.

  MODEL_DICTIONARY: 
  
 ** Starting from 1 **
models_String = ["SVM", "DT", "LDA", "KNN",
          "LogisticRegression", "GaussianNB", "RandomForest"]
  
  where models is a string of integers ex: 247 is DT, KNN, and Random Forest
  and where graphing and printing is an integer (0 = true, anything else= false) such that "true" means graph/print respectively.
  
