# AAPL_stock_price_prediction

## Abstract

Although many proposed stock price predicting models had great success in academia and industry, limitations can be experienced, i.e. they may fail to predict stock price accurately and in time. Some advanced data-driven methods have demonstrated stronger predictive capabilities, providing investors with the possibility of obtaining excess returns on investment. This project mainly investigates the importance of various features in predicting stock price with comprehensive data processing and analysis and conducts an empirical study to model the pattern of Apple Inc. (AAPL) stock price from April 2017 to May 2020 and predict its short-term trend. The experiment results show that features of AAPL stock trading volume, Nasdaq index and its trading volume, OCFPS and US retail sales can significantly contribute to improving the predictive capability of the model that uses historical stock price as the only predictor. The LSTM-based model achieves the lowest denormalized MSE at 0.06164 in the testing phase.

Main.py is the runnable script that can generate all the results shown in the report, i.e. data acqusition, preprocessing, EDA and inference. Some hits here:

1. The file reading directory (for dataset loading) is tested on Windows10 (x86) using Spyder as IDE, while uncertainty can be expected for running on OS or Linux or other IDEs. If errors encountered, please modify the path variables in main.py at line 34, 44 and 376 if uncommented. The data reading works on relative directory, try to excute main.py in the directory where it locates.

2. Two ways can be used to load dataset of Apple's financials: 
    
    1) by web scraping; 
    2) by locally loading, where 2) is a backup for off-line execution. 

Please read the comments in "Load and process financial datasets" cell and comment/uncomment corresponding chunks of code. Note that the scraped data may vary due to updates of the web.

3. Online datasets are independently stored in the Oracle Cloud using RESTful service. Please note that Oracle server sometimes break down with error code 500. If so, please try it later. Local datasets are stored in the "datasets" folder in CSV format.

4. Pre-trained LSTM models are saved in "model_files" folder, you can choose to skip the code cell configures model training.

Please check attached [report](https://github.com/JianqiaoMao/AAPL_stock_price_prediction/blob/master/ECEL0136%20Assignment_SN20041534_Report.pdf) for details.

## Environment

The whole project is developed in Python3.6. Please note that using other Python versions may lead to unknown errors. Required external libraries are shown below, where the recommended versions are also demonstrated:

numpy 1.19.1

pandas 1.1.3

requests 2.24.0

urllib3 1.25.10

ujson 4.0.1

seaborn 0.11.0

matplotlib 3.3.2

scipy 1.5.3

joblib 0.17.0

statsmodels 0.12.1

fbprophet 0.7.1

scikit-learn 0.23.2

keras 2.4.3

tensorflow-gpu 2.3.1 / Alternative: tensorflow (latest version)

Some other dependent libraries may be needed, please install as required.
