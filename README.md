# ECOC-Correlation-Testing
Programs for testing correlation between classifiers in ensemble learning via error correcting output codes.

Master_Tester_V2 Is most up to date, use on the commandline as follows:

./MasterTester_V2 <dataset> labelscol databegin dataend numclasses models graphing printing outfile
  
  MODEL_DICTIONARY: 
  
 ** Starting from 1 **
models_String = ["SVM", "DT", "LDA", "KNN",
          "LogisticRegression", "GaussianNB", "RandomForest"]
  
  where models is a string of integers ex: 247
  and where graphing and printing is an integer (0 = true, anything else= false) such that "true" means graph/print respectively.
  
