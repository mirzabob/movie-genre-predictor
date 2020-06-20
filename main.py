import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import os


f1=open('X.pckl','rb')
X=pickle.load(f1)
f2=open('Y.pckl','rb')
Y=pickle.load(f2)
f3=open('Genredict.pckl','rb')
Genre_ID_to_name=pickle.load(f3)
f4=open('movies_with_overviews.pckl','rb')
movies_with_overviews=pickle.load(f4)
f1.close()
f2.close()
f3.close()
f4.close()

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer()
X_tfidf=tfidf_transformer.fit_transform(X)
X_tfidf.shape

msk=np.random.rand(X_tfidf.shape[0])<0.8

X_train_tfidf=X_tfidf[msk]
X_test_tfidf=X_tfidf[~msk]
Y_train=Y[msk]
Y_test=Y[~msk]
positions=range(X_tfidf.shape[0])
test_movies=np.asarray(positions)[~msk]

#SVM Model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report

parameters = {'kernel':['linear'], 'C':[0.01, 0.1, 1.0]}
gridCV = GridSearchCV(SVC(class_weight='balanced'), parameters, scoring=make_scorer(f1_score, average='micro'))
classif = OneVsRestClassifier(gridCV)

classif.fit(X_train_tfidf, Y_train)

predstfidf=classif.predict(X_test_tfidf)

genre_list=sorted(list(Genre_ID_to_name.keys()))

predictions=[]
for i in range(X_test_tfidf.shape[0]):
    pred_genres=[]
    movie_label_scores=predstfidf[i]
    for j in range(19):
        if movie_label_scores[j]!=0:
            genre=Genre_ID_to_name[genre_list[j]]
            pred_genres.append(genre)
    predictions.append(pred_genres)


#f=open('classifier_svc','wb')
#pickle.dump(classif,f)
#f.close()
    

for i in range(X_test_tfidf.shape[0]):
    if i%50==0 and i!=0:
        print('MOVIE: ',movies_with_overviews[i]['title'],'\tPREDICTION: ',','.join(predictions[i]))





#Multinomial NB Model
from sklearn.naive_bayes import MultinomialNB
classifnb = OneVsRestClassifier(MultinomialNB())
classifnb.fit(X[msk].toarray(), Y_train)
predsnb=classifnb.predict(X[~msk].toarray())

#f2=open('classifer_nb','wb')
#pickle.dump(classifnb,f2)
#f2.close()

predictionsnb=[]
for i in range(X_test_tfidf.shape[0]):
    pred_genres=[]
    movie_label_scores=predsnb[i]
    for j in range(19):
        if movie_label_scores[j]!=0:
            genre=Genre_ID_to_name[genre_list[j]]
            pred_genres.append(genre)
    predictionsnb.append(pred_genres)


for i in range(X_test_tfidf.shape[0]):
    if i%50==0 and i!=0:
        print('MOVIE: ',movies_with_overviews[i]['title'],'\tPREDICTION: ',','.join(predictionsnb[i]))



#Comparing two models
def precision_recall(gt,preds):
    TP=0
    FP=0
    FN=0
    for t in gt:
        if t in preds:
            TP+=1
        else:
            FN+=1
    for p in preds:
        if p not in gt:
            FP+=1
    if TP+FP==0:
        precision=0
    else:
        precision=TP/float(TP+FP)
    if TP+FN==0:
        recall=0
    else:
        recall=TP/float(TP+FN)
    return precision,recall   

#For SVM
precs=[]
recs=[]
for i in range(len(test_movies)):
    pos=test_movies[i]
    test_movie=movies_with_overviews[pos]
    gtids=test_movie['genre_ids']
    gt=[]
    for g in gtids:
        g_name=Genre_ID_to_name[g]
        gt.append(g_name)
    a,b=precision_recall(gt,predictions[i])
    precs.append(a)
    recs.append(b)

        
print(np.mean(np.asarray(precs)),np.mean(np.asarray(recs)))

#For Multinomial NB
precs=[]
recs=[]
for i in range(len(test_movies)):
    if i%1==0:
        pos=test_movies[i]
        test_movie=movies_with_overviews[pos]
        gtids=test_movie['genre_ids']
        gt=[]
        for g in gtids:
            g_name=Genre_ID_to_name[g]
            gt.append(g_name)
        a,b=precision_recall(gt,predictionsnb[i])
        precs.append(a)
        recs.append(b)

print(np.mean(np.asarray(precs)),np.mean(np.asarray(recs)))


#Using Deep Learning for posters
poster_folder='posters_final/'

f=open('poster_movies.pckl','rb')
poster_movies=pickle.load(f)
f.close()

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
model = VGG16(weights='imagenet', include_top=False)


allnames=os.listdir(poster_folder)
imnames=[j for j in allnames if j.endswith('.jpg')]
feature_list=[]
genre_list=[]
file_order=[]
print("Starting extracting VGG features for scraped images. This will take time, Please be patient...")
print("Total images = ",len(imnames))
failed_files=[]
succesful_files=[]
i=0

for mov in poster_movies:
    i+=1
    mov_name=mov['original_title']
    mov_name1=mov_name.replace(':','/')
    poster_name=mov_name.replace(' ','_')+'.jpg'
    if poster_name in imnames:
        img_path=poster_folder+poster_name
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            succesful_files.append(poster_name)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model.predict(x)
            file_order.append(img_path)
            feature_list.append(features)
            genre_list.append(mov['genre_ids'])
            if np.max(np.asarray(feature_list))==0.0:
                print('problematic',i)
            if i%250==0 or i==1:
                print("Working on Image : ",i)
        except:
            failed_files.append(poster_name)
            continue
        
    else:
        continue
print("Done with all features, please pickle for future use!")

len(genre_list)
len(feature_list)

#list_pickled=(feature_list,file_order,failed_files,succesful_files,genre_list)
#f=open('posters_new_features.pckl','wb')
#pickle.dump(list_pickled,f)
#f.close()
#print("Features dumped to pickle file")

f=open('posters_new_features.pckl','rb')
list_pickled=pickle.load(f)
f.close()

(feature_list,files,failed,succesful,genre_list)=list_pickled

(a,b,c,d)=feature_list[0].shape
feature_size=a*b*c*d

np_features=np.zeros((len(feature_list),feature_size))
for i in range(len(feature_list)):
    feat=feature_list[i]
    reshaped_feat=feat.reshape(1,-1)
    np_features[i]=reshaped_feat

X=np_features

from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer()
Y=mlb.fit_transform(genre_list)

Y.shape
X.shape

#visual_problem_data=(X,Y)
#f=open('visual_problem_data_clean.pckl','wb')
#pickle.dump(visual_problem_data,f)
#f.close()


f=open('visual_problem_data_clean.pckl','rb')
visual_features=pickle.load(f)
f.close()

(X,Y)=visual_features

Y.shape
X.shape

mask = np.random.rand(len(X)) < 0.8

X_train=X[mask]
X_test=X[~mask]
Y_train=Y[mask]
Y_test=Y[~mask]

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

model_visual = Sequential([
    Dense(1024, input_shape=(25088,)),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(19),
    Activation('sigmoid'),
])
    
opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

model_visual.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_visual.fit(X_train, Y_train, epochs=10, batch_size=64,verbose=1)

model_visual.fit(X_train, Y_train, epochs=50, batch_size=64,verbose=0)

Y_preds=model_visual.predict(X_test)

sum(sum(Y_preds))

genre_list=sorted(list(Genre_ID_to_name.keys()))

precs=[]
recs=[]
for i in range(len(Y_preds)):
    row=Y_preds[i]
    gt_genres=Y_test[i]
    gt_genre_names=[]
    for j in range(19):
        if gt_genres[j]==1:
            gt_genre_names.append(Genre_ID_to_name[genre_list[j]])
    top_3=np.argsort(row)[-3:]
    predicted_genres=[]
    for genre in top_3:
        predicted_genres.append(Genre_ID_to_name[genre_list[genre]])
    (precision,recall)=precision_recall(gt_genre_names,predicted_genres)
    precs.append(precision)
    recs.append(recall)
    if i%50==0:
        print("Predicted: ",','.join(predicted_genres)," Actual: ",','.join(gt_genre_names))
        
print(np.mean(np.asarray(precs)),np.mean(np.asarray(recs)))