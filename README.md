# ECOC-Correlation-Testing
Programs for testing correlation between classifiers in ensemble learning via error correcting output codes.

Master_Tester_V2 Is most up to date, use on the commandline as follows:

python3 MasterTester_V2.py "datasets/pendigits.csv" -1 0 12 10 247 1 0 10 "./penDigits_cmd.txt"

Order of arguments: labelscolumn, databegin, dataend, numclasses, stringofmodels, graphing, printing, folds file_for_output

  MODEL_DICTIONARY: 
  
 ** Starting from 1 **
models_String = ["SVM", "DT", "LDA", "KNN",
          "LogisticRegression", "GaussianNB", "RandomForest"]
  
  where models is a string of integers ex: 247 is DT, KNN, and Random Forest
  and where graphing and printing is an integer (0 = true, anything else= false) such that "true" means graph/print respectively.
  
