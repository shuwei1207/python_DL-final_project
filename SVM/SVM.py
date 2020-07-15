from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

#Reshape the numpy array to fit in SVM
x_train = x_train.reshape((x_train.shape[0],img_size*img_size))
x_val = x_val.reshape((x_val.shape[0],img_size*img_size))
x_test = x_test.reshape((x_test.shape[0],img_size*img_size))


# Initialized the SVM with different kernel,such as rbf, linear, poly
svm = SVC(kernel = 'rbf',probability = True) 
svm.fit(x_train,y_train) # Training


# predict validation set
pred_x_val = svm.predict(x_val) 
accuracy_score(pred_x_val, y_val)


# predict testing set
pred_x_test = svm.predict(x_test) 
accuracy_score(pred_x_test, y_test)


# Plot the confusion matrix without normalization
titles_options = [("Confusion matrix, without normalization", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(svm, x_test, y_test,
                                 display_labels=[0,1],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


print(classification_report(y_test, pred_x_test, target_names = ['0', '1']))