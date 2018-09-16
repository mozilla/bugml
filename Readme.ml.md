In this repo there are python ML scripts for bugs classification,
csv files with bugs features (bug description, reporter, etc)
and script for downloading bugs data and save it to the csv file.

Prerequisites:

 - Python 3.6
 - Jupyter notebook
 - Python Scientific Development Environment (optional): Spyder or another.
 - Tensorflow
 - Keras
 - Numpy, scipy, pandas, sklearn, BeautifulSoup, gensim, nltk.


## ML in the file "bug_ml_tfidf.py".
### Description
    
Machine learning script for estimate the accuracy of the bugzilla bugs classifying
using different algorithms:
1. Naive Bayes
1. Logistic Regression
1. Support Vector Machine with Core
1. Random Forest
1. K-Nearest-Neighbors

Bug descriptions (text) converted to the vectors using tf-idf transformation

Other bug features (reporter, platform, OS) processed as category data

Input data for the script - csv file with examples of bugs (including bugs features - short description (summary), description (comments), reporter, reporter platform and OS)

Output script results:
1. Declassification models accuracy
1. Best models hyperparameters

### How to use
- Find file "bug_ml_tfidf.py".
- Open it with Spyder or Jupyter notebook.
- Install required python packages if they are not already installed.
- Launch script.

## ML in the file "BugsClassifier.ipynb".
### Description
With this notebook file you can:
 - download bugs data from mozilla
 - perform data munging on downloaded data, create and save datasets with different parameters
 - create, train neural network model to classify bugs by product - component labels
 - test neural networks on separated test data.
 - automatically search best hyperparameters for neural network models.
 - save information about models to history file
There are three models files in 'models' folder, which can be used with this script:
 - 'model_opt_py' (CNN)
 - 'model_rnn_py' (RNN)
 - 'model_rcnn_py' (CNN + LSTM)
They can be used for searching best hyperparameters and selecting best bugs classification model.
### How to use:
- Find file bugDataTest500.7z in 'data' folder, and extract it.
- Find file BugsClassifier.ipynb.
- Open it with jupyter notebook.
- Install required python packages if they are not already installed.
- After opening notebook, you can just sequentially run cells to performing these steps.
- Since we provided the data along with the script, you can skip downloading data step, it commented out in code.
