from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

x_train = x_train.reshape((x_train.shape[0],img_size*img_size))
x_val = x_val.reshape((x_val.shape[0],img_size*img_size))
x_test = x_test.reshape((x_test.shape[0],img_size*img_size))

#高斯貝氏分類器 GaussianNB
modelg=GaussianNB()
modelg.fit(x_train,y_train)
modelg.predict(x_test)


#多項式貝氏分類器 MultinomialNB
modelm=MultinomialNB()
modelm.fit(x_train,y_train)
modelm.predict(x_test)


#伯努力貝氏分類器 Bernoulli Naive Bayes
modelb = BernoulliNB(binarize=0.5)
modelb.fit(x_train,y_train)
modelb.predict(x_test)


print("高斯貝氏分類器")
print("test_accuracy",modelg.score(x_test,y_test))
print("val_accuracy",modelg.score(x_val,y_val))
# Plot the confusion matrix without normalization(GaussianNB)
titles_options = [("GaussianNB Confusion matrix, without normalization", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(modelg, x_test, y_test,
                                 display_labels=[0,1],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
print(classification_report(y_test, modelg.predict(x_test), target_names = ['0', '1']))


print("多項式貝氏分類器")
print("test_accuracy",modelm.score(x_test,y_test))
print("val_accuracy",modelm.score(x_val,y_val))
# Plot the confusion matrix without normalization(MultinomialNB)
titles_options = [("MultinomialNB Confusion matrix, without normalization", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(modelm, x_test, y_test,
                                 display_labels=[0,1],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
print(classification_report(y_test, modelm.predict(x_test), target_names = ['0', '1']))


print("伯努力貝氏分類器")
print("test_accuracy",modelb.score(x_test,y_test))
print("val_accuracy",modelb.score(x_val,y_val))
# Plot the confusion matrix without normalization(Bernoulli Naive Bayes)
titles_options = [("Bernoulli Naive Bayes Confusion matrix, without normalization", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(modelb, x_test, y_test,
                                 display_labels=[0,1],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
print(classification_report(y_test, modelb.predict(x_test), target_names = ['0', '1']))