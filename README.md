Hello. In this Project I have used two types of Machine Learning Algorithms for Human Recognition . For this project I have used the Images of 5 Currenct Bangladeshi Cricketers ( Which is a popular sport in subcontinent) . 

In the BanCricML.py code , I have used Support Vector Machines with HOG features. But before that , I used Good Old Fasioned Haar Cascade Classifier to detect the faces of the cricketers . Then I have Resized the Faces into same size. 
After that , I have extracted HOG Features and trained using Support Vector Machines with NuSVC. The 10 fold Cross Validation Accuracy is 93.8% while the Classifier Detected 15 out of 20 Test Images !


In the BanCricTF.py I have used the concept of Transfer Learning i.e. I have retrained the Inception model of Google Tensorflow using the Dataset . So I had to edit the retrain.py script from tensorflow examples. The link is given in the code. It's cross validation accuracy is 90% while the Test Accuracy is 87% . 


