Code Taken from Maxime Coriou and Athanasios Giannakopoulos from their work: "Everything Matters: A Robust Ensemble Architecture for End-to-End Text Classification" to establish a common-ground benchmark for the evaluation of multilingual churn detection. 
1. Select parameter in parameters.py

2. Change the dataset input path in run_model.py

3. Select the model to call in NeuralNets

4. If runing on Task C, change TestCallback.py (line 117) and EarlyStoppingByPatience.py (158) to evaluate with fscore

5. To save model uncomment lines in Earlystopping
