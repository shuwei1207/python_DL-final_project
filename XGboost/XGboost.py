from xgboost import XGBClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

#reshape x_train
x_train = x_train.reshape(len(x_train),img_size*img_size)


#train model
xgbc = XGBClassifier()
xgbc.fit(x_train, y_train)


#save image
digraph = xgb.to_graphviz(xgbc)
digraph.format = 'png'
digraph.view('xgb')


#predict test
x_test = x_test.reshape(len(x_test),img_size*img_size)
print("test_accuracy",xgbc.score(x_test,y_test))


#predict validation
x_val = x_val.reshape(len(x_val),img_size*img_size)
print("val_accuracy",xgbc.score(x_val,y_val))


#show confusion matrix
titles_options = [("Confusion matrix, itout normalization", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(xgbc, x_test, y_test,
                                display_labels=[0,1],
                                cmap=plt.cm.Blues,
                                normalize=normalize)
    disp.ax_.set_title(title)
    
    print(title)
    print(disp.confusion_matrix)
plt.show()


#show classification report
print(classification_report(y_test, xgbc.predict(x_test), target_names = ['0','1']))