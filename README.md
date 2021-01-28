# Adjusted-Cross-entropy-loss-function
Adjusted CrossEntropy loss function that integrates a penalty for the distance between the true class and the predicted one


The defined function adjust the loss of a multi-class classification NN for the distance between the true class y and the predicted class yhat.

Example of use : 
Assuming we want to categorize pictures of cat (class 0), dog (class 1), zebra (class 2), snake (class 3) and fish (class 4) 

If the true class is 1 (dog) AND that we consider it better to predict a cat (class 0) than a fich (class 4), 
the adjusted loss-function will improve the classification by the NN

