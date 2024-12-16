#   Author        : *** Fiona Seto ***
#   Last Modified : *** 12/7 ***

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import torch
import argparse
import helper
from sklearn import tree
from sklearn import metrics
from sklearn import neural_network
from sklearn import model_selection


def exp1(data, labels):
    """STUDENT CODE BELOW"""
    data=[99, 90, 37, 51, 70, 40, 23, 14,27,50,31,11,82,12]
    labels = [0,0,0,0,0,0,0,1,1,1,1,1,1,1]
    # importing other things needed for the model 
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    # splitting data for training and testing
    Data_train, Data_test, label_train, label_test = train_test_split(data, labels, test_size=.2, random_state=13300)
    print(Data_train.shape, Data_test.shape, label_train.shape, label_test.shape)
    # define model architecure, first test one had random state 13300 no max_depth
    model = tree.DecisionTreeClassifier(random_state = 13, max_depth=5) 
    model.fit(Data_train, label_train) # model training 
    predicted = model.predict(Data_test) # classifying test data
    accuracy = accuracy_score(predicted, label_test) # checking accuracy 
    print(f'Accuracy: {accuracy}')

    #saving plot, imports for trees 
    from sklearn.tree import plot_tree
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_curve 
    #tree
    plt.figure(figsize=(30,20)) # make model large enough to see words 
    plot_tree(model, filled=True) # fill to give color 
    plt.savefig("tree_img.png")
    #feat
    fImportance = model.feature_importances_ # getting data 
    plt.figure(figsize=(10,6)) #setting size 
    plt.barh(data.columns, fImportance) # plotting
    plt.tight_layout() # making plot fit on png 
    plt.savefig("feature_importance.png")
    #conf
    label_pred = model.predict(Data_test) # classifying test data
    conMatrix = confusion_matrix(label_test, label_pred) # making matrix
    plt.figure(figsize=(10,6)) # setting size 
    sns.heatmap(conMatrix) #plotting 
    plt.savefig("confusion_matrix.png")
    #ROC
    l_prob = model.predict_proba(Data_test)[:, 1] #gettingn predicted probability for positive/ malicous data
    fpr, tpr, thresholds = roc_curve(label_test, l_prob) # computing false and true positive rates 
    plt.figure(figsize=(10, 6)) # setting size
    plt.plot(fpr, tpr) #plotting 
    plt.xticks([i * 0.05 for i in range(21)]) #specifying ticks to be more specific 
    plt.savefig("roc_curve.png")
    
    """STUDENT CODE ABOVE"""
    return model


def exp2(data, labels):
    """STUDENT CODE BELOW"""
    # importing things for model 
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    #split data
    Data_train, Data_test, label_train, label_test = train_test_split(data, labels, test_size=.2, random_state=13300)
    print(Data_train.shape, Data_test.shape, label_train.shape, label_test.shape)
    # define model architecure, tested one had random state 13300 no max_depth
    model = tree.DecisionTreeClassifier(random_state = 13, max_depth=20) 
    model.fit(Data_train, label_train) #model training 
    predicted = model.predict(Data_test) #model classfying 
    accuracy = accuracy_score(predicted, label_test) #model accuracy 
    print(f'Accuracy: {accuracy}')

    #plots and their imports 
    from sklearn.metrics import confusion_matrix
    #tree
    from sklearn.tree import plot_tree
    from sklearn.metrics import confusion_matrix
    plt.figure(figsize=(60,45)) #setting size
    plot_tree(model, filled=True, fontsize=5) #changing font size for readability 
    plt.savefig("tree_img2.png")
    #feat
    fImportance = model.feature_importances_ #getting important features 
    plt.figure(figsize=(10,6)) 
    plt.barh(data.columns, fImportance) #plotting 
    plt.tight_layout() #making sure the plot fits 
    plt.savefig("feature_importance2.png")
    #conf
    label_pred = model.predict(Data_test) #model classifying 
    conMatrix = confusion_matrix(label_test, label_pred) #making matrix 
    plt.figure(figsize=(10,6)) 
    sns.heatmap(conMatrix) #plotting
    plt.savefig("confusion_matrix2.png")
    #test info
    report = classification_report(label_test,predicted)
    print(f'Classification Report:\n;, {report}')
    """STUDENT CODE ABOVE"""
    return model


def exp3(data, labels):
    """STUDENT CODE BELOW"""
    #imports needed
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    #split data
    Data_train, Data_test, label_train, label_test = train_test_split(data, labels, test_size=.2, random_state=50)
    #scaler = StandardScaler()
    #Data_train = scaler.fit_transform(Data_train)
    #Data_test = scaler.transform(Data_test)
    # define model architecture
    model = neural_network.MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=50, early_stopping=True, alpha=1e-3, activation = 'relu', solver='adam')
    
    model.fit(Data_train, label_train) #training 

    predicted = model.predict(Data_test) #classifying

    accuracy = accuracy_score(predicted, label_test) #accuracy of prediction 
    print(f'Accuracy: {accuracy}')
    #plot imports
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_curve 
    #loss curve
    plt.plot(model.loss_curve_)
    plt.savefig("loss_curve3.png")
    #conf
    label_pred = model.predict(Data_test) #classifying
    conMatrix = confusion_matrix(label_test, label_pred) # making matrix
    plt.figure(figsize=(10,6)) #set size
    sns.heatmap(conMatrix) #plotting
    plt.savefig("confusion_matrix3.png")

    #ROC
    l_prob = model.predict_proba(Data_test)[:, 1]#getting predicted probability
    fpr, tpr, thresholds = roc_curve(label_test, l_prob) #getting false and true positive rate
    plt.figure(figsize=(10, 6)) #set size
    plt.plot(fpr, tpr) #ploting
    plt.xticks([i * 0.05 for i in range(21)]) #specifying x axis 
    plt.savefig("roc_curve3.png")
    print("Thresholds:", thresholds)
    # getting difference between tpr fpr 
    tf_diff = tpr - fpr

    # finding optimal threshold 
    optimal_idx = np.argmax(tf_diff)
    optimal_threshold = thresholds[optimal_idx]
    print("Optimal Threshold:", optimal_threshold)
    # info 
    report = classification_report(label_test,predicted)
    print(f'Classification Report:\n;, {report}')
    """STUDENT CODE ABOVE"""
    return model


def exp4(data, labels):
    """STUDENT CODE BELOW"""
    #imports for model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    # splitting inputs 
    Data_train, Data_test, label_train, label_test = train_test_split(data, labels, test_size=.2, random_state=50)
    #scaler = StandardScaler()
    #Data_train = scaler.fit_transform(Data_train)
    #Data_test = scaler.transform(Data_test)
    # define model architecture
    model = neural_network.MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=50, early_stopping=True, alpha=1e-3, activation = 'relu', solver='adam')
    
    model.fit(Data_train, label_train) #training 

    predicted = model.predict(Data_test) #classifying

    accuracy = accuracy_score(predicted, label_test) #accuracy of prediction 
    print(f'Accuracy: {accuracy}')
    # import plot 
    from sklearn.metrics import confusion_matrix
    #loss curve
    plt.plot(model.loss_curve_)
    plt.savefig("loss_curve4.png")
    #conf
    label_pred = model.predict(Data_test) #classifying
    conMatrix = confusion_matrix(label_test, label_pred) #getting matrix
    plt.figure(figsize=(10,6)) #set size
    sns.heatmap(conMatrix) # plotting 
    plt.savefig("confusion_matrix4.png")

    #test information 
    report = classification_report(label_test,predicted)
    print(f'Classification Report:\n;, {report}')
    """STUDENT CODE ABOVE"""
    return model


def exp5(data, labels):
    """STUDENT CODE BELOW"""
    # convert data to pytorch dataset
    dataset = helper.convert_to_pytorch_dataset(data, labels)
    # define model architecture
    model = torch.nn.Sequential(
        torch.nn.Linear(40, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 5),
    )
    """STUDENT CODE ABOVE"""
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=1, help="Experiment number")
    args = parser.parse_args()
    save_name = f"exp{args.exp}_model" + (".pt" if args.exp == 5 else ".pkl")
    if args.exp == 1:
        model = exp1(*helper.load_dataset(multiclass=False, normalize=False))
    elif args.exp == 2:
        model = exp2(*helper.load_dataset(multiclass=True, normalize=False))
    elif args.exp == 3:
        model = exp3(*helper.load_dataset(multiclass=False, normalize=True))
    elif args.exp == 4:
        model = exp4(*helper.load_dataset(multiclass=True, normalize=True))
    elif args.exp == 5:
        model = exp5(*helper.load_dataset(multiclass=True, normalize=True))
    else:
        print("Invalid experiment number")
    helper.save_model(model, save_name)
