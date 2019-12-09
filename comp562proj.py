
print("test")
srcPath='images_5_classes'
import numpy as np
import gzip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas
from os import listdir
from os.path import isfile, join
import collections
import time
import sklearn.mixture
from numpy import array
import sys
import math
import os
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from mpl_toolkits.mplot3d import axes3d
from matplotlib.cbook import get_sample_data
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d import proj3d
import pylab
from collections import Counter
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
import random
import seaborn as sns
from sklearn import preprocessing
'''
try:
    import cPickle as pickle
    kwargs = {}
except:
    import _pickle as pickle
    kwargs = {'encoding':'bytes'}
    
theta = [1.0/5.0]*5 # this makes an array of length 5 with entries all being 1.0/5.0
print ("Theta = ", theta)
state_space = [1,2,3,4,5]
x = np.random.choice(state_space, size=100, p=theta)
print ('x = ', x)
#plt.hist(x, bins=[0.5,1.5,2.5,3.5,4.5,5.5], align='mid', edgecolor='black')
#plt.xlabel('X')
#plt.ylabel('count')
#plt.show()
mus =    [1.0,10.0,-7.0]
sigmas = [1.0, 1.0, 2.0]
for (mu,sigma) in zip(mus,sigmas):
    x = np.random.normal(mu,sigma,2000)
    print ("Mean: " + str(np.mean(x)) + " Standard Deviation: " + str(np.std(x)))
    #plt.xlim(-15,15)
    #plt.hist(x,40,normed=True,label='mu:'+str(mu)+' sigma:'+str(sigma))

#plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
#plt.show()
features, labels, sample_ids, label_names =  pickle.load( gzip.open( 'hw2_data.pgz', 'rb' ), **kwargs )

np.random.seed(12345)
N = features.shape[0]
arr = np.arange(N)
np.random.shuffle(arr)
train_num = int(round(N*0.8))
test_num = features.shape[0]-train_num

train_subset = arr[:train_num]
train_features = features[train_subset,:]
train_labels = labels[train_subset]
train_sample_ids = [sample_ids[i] for i in train_subset]


test_subset = arr[train_num:]
test_features = features[test_subset,:]
test_labels = labels[test_subset]
test_sample_ids = [sample_ids[i] for i in test_subset]
label_set = np.unique(test_labels)
#fIdx = maxErrNameMap[(label_set[rIdx], label_set[cIdx])] 
#plt.axis('off')
if False:
    for i in range(5):
            rIdx = i // 5
            cIdx = i % 5    
            fIdx=label_set[rIdx]
            folderName = label_names[test_labels[fIdx]]            
            fID = test_sample_ids[fIdx]
            img=mpimg.imread( srcPath + '/' + folderName.decode('utf-8') + '/' + fID.decode('utf-8') + '.jpg')
            print("IMAGE DATA",len(img.flatten()),len(img),len(img[0]),len(img[0][0]))
            #gives RGB coords
            plt.axis('off') 
            plt.xticks()
            plt.yticks()
            plt.xlabel("")
            plt.ylabel("")
            plt.subplots_adjust(wspace=0, hspace=0)
            print("showing",(srcPath + '/' + folderName.decode('utf-8') + '/' + fID.decode('utf-8') + '.jpg'))
            plt.imshow(img)
            plt.show()
#plt.show()
        
'''
picZoom=0.3
paintingMetadata=pandas.read_csv("___paintingdata/all_data_info.csv")
print("**********************painting metadata",'\n\n',paintingMetadata,'\n\n',paintingMetadata['new_filename'],'\n\n',
      paintingMetadata.columns,'\n\n',paintingMetadata.columns[0],'\n\n',
      paintingMetadata.loc[paintingMetadata['new_filename'] == '1.jpg'].to_string(),'\n\n',
      paintingMetadata.loc[paintingMetadata['new_filename'] == '1.jpg']['artist'].to_string(),'\n\n',
      (paintingMetadata.loc[paintingMetadata['new_filename'] == '1.jpg']['artist']=='M.C. Escher').to_string(),'\n *************************************************',
      len(paintingMetadata.loc[paintingMetadata['style'] == 'Realism']))


paintingPicturesPath='___paintingdata/train'
paintingPictures = [f for f in listdir(paintingPicturesPath) if isfile(join(paintingPicturesPath, f))]
print(paintingPictures[0:10],'\n',len(paintingPictures))

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

paintingImageFeatures=[]
#np.append(paintingImageFeatures,np.asarray(np.ones(shape=(1,10))))
test1=[1,2,3,4]
test2=[5,6,7,8]
test3=np.stack([test1,test2])
#print("|||||||||||||||||||||||||||||||||||",paintingImageFeatures,'\npppppppppppppppppppp',test3)
paintingImageLabels=[]
#paintingImageLabelsDigits=[]
'''average color rgb
average color in grayscale
contrast (difference between min and max pixel in grayscale)
darkest color grayscale
brightest color grayscale
'''
def luminance(rgb):
    return (rgb[0]*0.3)+(rgb[1]*0.59)+(rgb[2]*0.11)
#paintingImageFeatures.append([1,2,3,4])
#paintingImageFeatures.append([5,6,7,8])
#print(paintingImageFeatures,paintingImageFeatures[0])
#preprocess image
#******************255 is darker
#******************luminance and grayscale value are the same
#make sure to convert to 0-1.0
'''intersection over union
get # of items that are in both clusters. the highest score of all of them is the score of the method
'''
skips=[5,5]
showImages=False
i=0
imagesMarkedForDeletion=[]
rangeOfImagesToProcess=200
numOfImagesToShowOnGraph=75
visualsSkip=int(rangeOfImagesToProcess/numOfImagesToShowOnGraph)
rangeOfImagesToShow=rangeOfImagesToProcess
start_time = time.time()
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",type(4) is int)
print("++++++++++++++++++++++++++++++++++",collections.Counter(paintingMetadata['style'])['Impressionism'])
imagesForOutput=[] #this is to avoid re-imreading

#sys.exit(0)
featureArraySplit=[[],[],[]]
for painting in paintingPictures[0:rangeOfImagesToProcess]:
    thisPainting=mpimg.imread(paintingPicturesPath+'/'+painting)
    thisPaintingDataframe=paintingMetadata.loc[paintingMetadata['new_filename']== painting]
    imageStyle=thisPaintingDataframe['style'].values[0]
    
    if showImages:
        plt.axis('off') 
        plt.xticks()
        plt.yticks()
        plt.xlabel("")
        plt.ylabel("")
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(thisPainting)
        plt.show()
    limits=[len(thisPainting),len(thisPainting[0])]
    numPixels=limits[0]*limits[1]
    print("____________________________this pic is",i,painting,len(thisPainting),thisPainting[0][0])#,type(thisPainting[0][0]),imageStyle,imageStyle=="Baroque",type(imageStyle))
    #print(paintingImageLabels.count('Baroque'),'\n')#,collections.Counter(paintingImageLabels))
    #print("IMAGE STYLE IS",imageStyle)
    if type(thisPainting[0][0]) is not np.ndarray or str(imageStyle)=='nan' or imageStyle==None:
        pass
        imagesMarkedForDeletion.append(painting)
        print("*****delete",painting)
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^not an image?",painting)
    else:
        imagesForOutput.append(thisPainting)
        paintingImageLabels.append(imageStyle)
        averageRGB=[0,0,0]
        averageColorGrayscale=0
        averageSaturation=0
        darkestColorInRGB=[0,0,0] #I can get this by converting to grayscale
        lightestColorInRGB=[1.0,1.0,1.0]
        darkestColorInGrayscale=0
        lightestColorInGrayscale=1.0
        averageDarkness=0
        numDarkPixels=0
        averageBrightness=0
        numBrightPixels=0
        
        for thisPixelX in range(0,limits[0],skips[0]):
            for thisPixelY in range(0,limits[1],skips[1]):
                thisPixelRGB=(thisPainting[thisPixelX][thisPixelY][0:3])/255.0
                thisPixelGrayscale=luminance(thisPixelRGB)
                averageRGB+=thisPixelRGB
                averageColorGrayscale+=thisPixelGrayscale
                minPixel=min(thisPixelRGB[0],thisPixelRGB[1],thisPixelRGB[2])
                maxPixel=max(thisPixelRGB[0],thisPixelRGB[1],thisPixelRGB[2])
                saturation=(maxPixel-minPixel)/maxPixel
                if not math.isnan(saturation):
                    averageSaturation+=saturation
                #print("-----------",minPixel,maxPixel,averageSaturation)
                if thisPixelGrayscale<lightestColorInGrayscale:
                    lightestColorInGrayscale=round(thisPixelGrayscale,4)
                    lightestColorInRGB=[round(thisPixelRGB[0],4),round(thisPixelRGB[1],4),round(thisPixelRGB[2],4)]
                if thisPixelGrayscale>darkestColorInGrayscale:
                    darkestColorInGrayscale=round(thisPixelGrayscale,4)
                    darkestColorInRGB=[round(thisPixelRGB[0],4),round(thisPixelRGB[1],4),round(thisPixelRGB[2],4)]
                if thisPixelGrayscale>=0.5:
                    averageDarkness+=2.0*(thisPixelGrayscale-0.5)
                    numDarkPixels+=1
                if thisPixelGrayscale<0.5:
                    averageBrightness+=2.0*(thisPixelGrayscale+0.5)
                    numBrightPixels+=1
                #print("this pixel rgb",thisPixelRGB,averageRGB)#,thisPainting[thisPixelX][thisPixelY])
        averageRGB=[round(averageRGB[0]/numPixels,4),round(averageRGB[1]/numPixels,4),round(averageRGB[2]/numPixels,4)]
        averageColorGrayscale=1.0-round(averageColorGrayscale/numPixels,6)
        
        
        averageBrightness=round(averageBrightness/numBrightPixels,4) if numBrightPixels>0 else 0.5
        averageDarkness=round(averageDarkness/numDarkPixels,4) if numDarkPixels>0 else 0.5
        averageSaturation=round(averageSaturation/numPixels,6)
        
        if math.isnan(averageSaturation):
            print("333333333333333333333333",averageSaturation)
            sys.exit()
        
        paintingImageFeatures.append([#averageRGB[0],averageRGB[1],averageRGB[2],
                                      averageColorGrayscale,
                                      #darkestColorInRGB[0],darkestColorInRGB[1],darkestColorInRGB[2],
                                      #lightestColorInRGB[0],lightestColorInRGB[1],lightestColorInRGB[2],
                                      #darkestColorInGrayscale,
                                      #lightestColorInGrayscale,
                                      #round(darkestColorInGrayscale-lightestColorInGrayscale,4),
                                      #averageDarkness,
                                      #averageBrightness,
                                      averageSaturation,
                                      round(averageDarkness-averageBrightness,6)]),
        featureArraySplit[0].append(averageColorGrayscale)
        featureArraySplit[1].append(averageSaturation)    
        featureArraySplit[2].append(round(averageDarkness-averageBrightness,6))    
    #print("333333333333333333333333",averageSaturation)
        i+=1
        #print("image features",i,paintingImageFeatures[len(paintingImageFeatures)-1], paintingImageLabels[len(paintingImageFeatures)-1])

allPaintingStyles=np.unique(paintingImageLabels)
countOfEachPaintingStyle=collections.Counter(paintingImageLabels)

##############################allPaintingGenres=np.unique(paintingMetadata['genre'])
#countOfEachPaintingGenre=collections.Counter(paintingMetadata['genre'])

print("-------------------------------------------------------------------------------------------numPicsToDelete",len(imagesMarkedForDeletion))

for image in imagesMarkedForDeletion:
    os.remove(paintingPicturesPath+'/'+image)
print(paintingImageLabels.count('Baroque'),'\n processed in ',time.time() - start_time,'\n',collections.Counter(paintingImageLabels))
'''
fig = plt.figure()
ax2 = fig.add_subplot(111)
#print("REDUCED DATA",reducedData[:, 2])
ax2.scatter(featureArraySplit[0],np.zeros(len(featureArraySplit[0])),c='r',marker='o',cmap='gist_rainbow');
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')
fig.suptitle('avg color grayscale', fontsize=16)
#print(featureMatrix,'\n',featureMatrix[0],'\n',featureMatrix[1],'\n\n\n\n\n',featureMatrix[:, 0])
#for cluster in range(numClusters):
    #print('cluster: ', cluster,np.where(labels == cluster))
    #print(paintingImageLabels[np.where(labels == cluster)])
annotations=[]
textAnnotations=[]

print("FEATURE ARRAY",featureArraySplit[0])

for i, txt in enumerate(paintingImageLabels):
    im = OffsetImage(mpimg.imread(paintingPicturesPath+'/'+paintingPictures[i]), zoom=0.1)
    im.image.axes = ax2
    
    ab = AnnotationBbox(im, xy=(featureArraySplit[0][i],0),
                        xybox=(0, 0),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.0,
                        arrowprops=dict(arrowstyle="->"))
    ab.set_zorder(-10)
    annotations.append(ab)
    
    ax2.add_artist(ab)
    
    #ann=ax.annotate(
        #txt, 
        #xy=(x2, y2), size=5,
        #bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.2))
    ann=Annotation(txt, 
        xy=(featureArraySplit[0], featureArraySplit[0]), size=5,
        bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.2))
    ax2.add_artist(ann)
    textAnnotations.append(ann)

plt.show()





fig2 = plt.figure()
ax3 = fig2.add_subplot(111)
#print("REDUCED DATA",reducedData[:, 2])
ax3.scatter(featureArraySplit[1],np.zeros(len(featureArraySplit[0])),c='b',marker='o',cmap='gist_rainbow');
ax3.set_xlabel('x axis')
ax3.set_ylabel('y axis')
#print(featureMatrix,'\n',featureMatrix[0],'\n',featureMatrix[1],'\n\n\n\n\n',featureMatrix[:, 0])
#for cluster in range(numClusters):
    #print('cluster: ', cluster,np.where(labels == cluster))
    #print(paintingImageLabels[np.where(labels == cluster)])
annotations=[]
textAnnotations=[]
fig2.suptitle('avg saturation', fontsize=16)

for i, txt in enumerate(paintingImageLabels):
    im = OffsetImage(mpimg.imread(paintingPicturesPath+'/'+paintingPictures[i]), zoom=0.1)
    im.image.axes = ax3
    
    ab = AnnotationBbox(im, xy=(featureArraySplit[1][i],0),
                        xybox=(0, 0),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.0,
                        arrowprops=dict(arrowstyle="->"))
    ab.set_zorder(-10)
    annotations.append(ab)
    
    ax3.add_artist(ab)
    
    #ann=ax.annotate(
        #txt, 
        #xy=(x2, y2), size=5,
        #bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.2))
    ann=Annotation(txt, 
        xy=(featureArraySplit[0], featureArraySplit[0]), size=5,
        bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.2))
    ax2.add_artist(ann)
    textAnnotations.append(ann)

plt.show()




fig3 = plt.figure()
ax4 = fig3.add_subplot(111)
#print("REDUCED DATA",reducedData[:, 2])
ax4.scatter(featureArraySplit[1],np.zeros(len(featureArraySplit[0])),c='g',marker='o',cmap='gist_rainbow');
ax4.set_xlabel('x axis')
ax4.set_ylabel('y axis')
#print(featureMatrix,'\n',featureMatrix[0],'\n',featureMatrix[1],'\n\n\n\n\n',featureMatrix[:, 0])
#for cluster in range(numClusters):
    #print('cluster: ', cluster,np.where(labels == cluster))
    #print(paintingImageLabels[np.where(labels == cluster)])
annotations=[]
textAnnotations=[]
fig3.suptitle('avg luminance delta', fontsize=16)

for i, txt in enumerate(paintingImageLabels):
    im = OffsetImage(mpimg.imread(paintingPicturesPath+'/'+paintingPictures[i]), zoom=0.1)
    im.image.axes = ax4
    
    ab = AnnotationBbox(im, xy=(featureArraySplit[1][i],0),
                        xybox=(0, 0),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.0,
                        arrowprops=dict(arrowstyle="->"))
    ab.set_zorder(-10)
    annotations.append(ab)
    
    ax4.add_artist(ab)
    
    #ann=ax.annotate(
        #txt, 
        #xy=(x2, y2), size=5,
        #bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.2))
    ann=Annotation(txt, 
        xy=(featureArraySplit[0], featureArraySplit[0]), size=5,
        bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.2))
    ax2.add_artist(ann)
    textAnnotations.append(ann)

plt.show()
'''





fig4 = plt.figure()
ax5 = fig4.add_subplot(111, projection='3d')
#print("REDUCED DATA",reducedData[:, 2])
ax5.scatter(featureArraySplit[0][::visualsSkip], featureArraySplit[1][::visualsSkip], featureArraySplit[2][::visualsSkip],c=(0,0,0,0),marker='.',cmap='gist_rainbow');
ax5.set_xlabel('grayscale')
ax5.set_ylabel('saturation')
ax5.set_zlabel('luminance')
#print(featureMatrix,'\n',featureMatrix[0],'\n',featureMatrix[1],'\n\n\n\n\n',featureMatrix[:, 0])
#for cluster in range(numClusters):
    #print('cluster: ', cluster,np.where(labels == cluster))
    #print(paintingImageLabels[np.where(labels == cluster)])
annotations2=[]
textAnnotations2=[]

for i, txt in enumerate(paintingImageLabels[::visualsSkip]):
    x2, y2, _ = proj3d.proj_transform(featureArraySplit[0][i*visualsSkip],featureArraySplit[1][i*visualsSkip],featureArraySplit[2][i*visualsSkip], ax5.get_proj())
    
    #annotationPosition=(random.randrange(0,1)*(1 if random.randint(0,1)>0 else -1),
                        #random.randrange(0,1)*(1 if random.randint(0,1)>0 else -1))
    
    
    
    im = OffsetImage(imagesForOutput[i*visualsSkip], zoom=picZoom)
    im.image.axes = ax5
    
    ab = AnnotationBbox(im, (x2,y2),
                        xybox=(0, 0),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.0,
                        arrowprops=dict(arrowstyle="->"))
    ab.set_zorder(-10)
    annotations2.append(ab)
    
    ax5.add_artist(ab)
    
    #ann=ax.annotate(
        #txt, 
        #xy=(x2, y2), size=5,
        #bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.2))
    ann=Annotation(txt, 
        xy=(x2, y2), size=5,
        bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.9))
    ax5.add_artist(ann)
    textAnnotations2.append(ann)





def update_position(e):
    for i in range(0,len(paintingImageLabels[::visualsSkip])):
        x2, y2, _ = proj3d.proj_transform(featureArraySplit[0][i*visualsSkip],featureArraySplit[1][i*visualsSkip],featureArraySplit[2][i*visualsSkip], ax5.get_proj())
        annotations2[i].xy = x2,y2
        annotations2[i].update_positions(fig4.canvas.renderer)        
        #textAnnotations2[i].set_position((x2,y2))
    fig4.canvas.draw()
#set_size(100,100,ax)
fig4.canvas.mpl_connect('button_release_event', update_position)

fig4.tight_layout()

plt.show()







for image in imagesMarkedForDeletion:
    os.remove(paintingPicturesPath+'/'+image)
print(paintingImageLabels.count('Baroque'),'\n processed in ',time.time() - start_time,'\n',collections.Counter(paintingImageLabels))
#print("{{{{{{{{____",paintingImageFeatures)
featureMatrix=preprocessing.scale(np.stack(paintingImageFeatures))
#print("[[[[[[[[[[[[[[[[____",featureMatrix)
#featureMatrix=np.asmatrix(paintingImageFeatures)
#print(featureMatrix)

'''class Annotation3D(Annotation):
    #Annotate the point xyz with text s

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)
        
        

def annotate3D(ax, s, *args, **kwargs):
    #add anotation text s to to Axes3d ax

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)'''
    
    
    
    
    
    
    
    
print("this many features=",len(paintingImageFeatures[0]),'___this many labels',len(collections.Counter(paintingImageLabels)))
#print(gmm)
numClusters=len(collections.Counter(paintingImageLabels))
numClusters=4
numPCAComponents=3 #so I can do 3d render!
from sklearn import preprocessing
reducedData=PCA(n_components=numPCAComponents).fit_transform(featureMatrix)
#reducedData=preprocessing.StandardScaler().fit_transform(PCA(n_components=numPCAComponents).fit_transform(featureMatrix))
#reducedData=pandas.DataFrame(reducedData,columns=featureMatrix.columns)
reducedData=featureMatrix


gmm=sklearn.mixture.GaussianMixture(n_components=numClusters,covariance_type='full',verbose=0,n_init=20,max_iter=1000).fit(reducedData)
labels2=gmm.predict(reducedData)
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn import datasets


clustersBrokenUp=[] #this just splits the labels array into smaller arrays so I can do mode
for cluster in range(numClusters):
    clustersBrokenUp.append([])

uniqueLabels=np.unique(paintingImageLabels).tolist()
intLabels=[]
for labelIndex in range(len(paintingImageLabels)):
    #print("LABELELELEL",paintingImageLabels[labelIndex],str(paintingImageLabels[labelIndex])=='nan',paintingImageLabels[labelIndex]==None)
    intLabels.append(uniqueLabels.index(paintingImageLabels[labelIndex]))
    clustersBrokenUp[labels2[labelIndex]].append(uniqueLabels.index(paintingImageLabels[labelIndex]))
#print("FINAL******\n",intLabels,'\n',labels2,'\n',clustersBrokenUp)


totalInSubClusterArray=0
#sanity check
for subcluster in range(len(clustersBrokenUp)):
    totalInSubClusterArray+=len(clustersBrokenUp[subcluster])
print("SANITTY",totalInSubClusterArray,len(paintingImageLabels))
'''
for cluster in range(numClusters):
    cat2=(labels2==cluster)
    cat=(labels==cluster)
    labels2[cat2]=mode(intLabels[cat2])
    labels[cat]=mode(intLabels[cat])'''
    
'''

figure out the mode genre label of each cluster

'''
from collections import Counter
desiredLabelForEachCluster=[]
for cluster in range(numClusters):
    c=Counter(clustersBrokenUp[cluster])
    desiredLabelForEachCluster.append(c.most_common(1)[0][0])
print("DESIRED",desiredLabelForEachCluster)

c1=Counter(intLabels)
print("***********c1\n",c1.keys(),'\n',c1.values(),'\n^^^^^^^^^^^^^^^^^^^^^^^\n')



accuracyOfEachCluster=[]
accuracyOfDesiredInEachCluster=[]

labelsCopied=labels2.copy()

for cluster in range(numClusters):
    howManyOfDesiredLabelShouldBeInThisCluster=intLabels.count(desiredLabelForEachCluster[cluster])
    for l in range(len(labelsCopied)):
        if labelsCopied[l] == cluster:
            labelsCopied[l]=desiredLabelForEachCluster[cluster]
    #howManyOfDesiredLabelAreInThisCluster
    numCorrectInThisCluster=0
    for e in clustersBrokenUp[cluster]:
        if e == desiredLabelForEachCluster[cluster]:
            numCorrectInThisCluster+=1
    accuracyOfEachCluster.append(numCorrectInThisCluster/len(clustersBrokenUp[cluster]))
    accuracyOfDesiredInEachCluster.append(numCorrectInThisCluster/howManyOfDesiredLabelShouldBeInThisCluster)
    
# i can also measure how many of the elements in ALL of the labels with the desired label were in that cluster
print("acc",accuracyOfEachCluster,'\n\n\n',accuracyOfDesiredInEachCluster,'\n\n\n',sum(accuracyOfEachCluster)/len(accuracyOfEachCluster),
      sum(accuracyOfDesiredInEachCluster)/len(accuracyOfDesiredInEachCluster),'\n\n\n%%%%%%%%%%%%%%%%%')


acc2=accuracy_score(intLabels,labelsCopied)
print("accuracies",acc2)

###############################################################measure accuracy: for each cluster, how many are in the same group
#for each genre, count number of paintings in that genre in each cluster. do cluster with greatest size / total number of paintings in that genre

'''
allPaintingStyles=np.unique(paintingMetadata['style'])
countOfEachPaintingStyle=collections.Counter(paintingMetadata['style'])

allPaintingGenres=np.unique(paintingMetadata['genre'])
countOfEachPaintingGenre=collections.Counter(paintingMetadata['genre'])

'''

gmmaccfig = plt.figure()
gmmaccax = gmmaccfig.add_subplot(111)
gmmaccax.set_title("GMM ACCURACIES")

accuracyForEachStyle=[]
styleText=[]
styleIndex=[]
accuracyText=[]
for style in range(len(allPaintingStyles)):
    howManyOfThisStyleThereAre=countOfEachPaintingStyle[allPaintingStyles[style]]
    howManyOfThisStyleAreInEachCluster=[]
    for cluster in range(numClusters):
        howManyOfThisStyleAreInEachCluster.append(0)
    #print("$",howManyOfThisStyleAreInEachCluster)
    for thisLabel in range(len(labels2)):
        howManyOfThisStyleAreInEachCluster[labels2[thisLabel]]+=1
    
    maxNumberOfThisStyleInAnyCluster=-1
    for cluster in range(numClusters):
        if howManyOfThisStyleAreInEachCluster[cluster]>maxNumberOfThisStyleInAnyCluster:
            maxNumberOfThisStyleInAnyCluster=howManyOfThisStyleAreInEachCluster[cluster]
    #print("----------------accuracy for this style:",allPaintingStyles[style],round(howManyOfThisStyleThereAre/maxNumberOfThisStyleInAnyCluster,4),"n=",howManyOfThisStyleThereAre)
    accuracyForEachStyle.append((round(howManyOfThisStyleThereAre/maxNumberOfThisStyleInAnyCluster,4),allPaintingStyles[style],howManyOfThisStyleThereAre,style))
    styleText.append(allPaintingStyles[style])
    styleIndex.append(style)
    accuracyText.append(round(howManyOfThisStyleThereAre/maxNumberOfThisStyleInAnyCluster,4))
fontsize1=8
sortedVals=sorted(accuracyForEachStyle, key=lambda x: x[0],reverse=True)
#res_list = [x[0] for x in accuracyForEachStyle]
#sortedVals=sorted()
bars=gmmaccax.barh(np.arange(len(sortedVals))*2,[x[0] for x in sortedVals])
print('gmm ((((((((((((((((((((((',sortedVals)
#gmmaccax.set_xticks([x[3] for x in sortedVals])
#gmmaccax.set_xticklabels([x[1] for x in sortedVals])
plt.yticks(np.arange(len(sortedVals))*2, [x[1] for x in sortedVals], fontsize=fontsize1)
#gmmaccax.set_xticklabels([x[1] for x in sortedVals])
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in range(len(rects)):
        height = rects[rect].get_width()
        gmmaccax.annotate('{}'.format('n='+str(sortedVals[rect][2])),
                    xy=(height,rects[rect].get_y() - rects[rect].get_height() / 2),
                    xytext=(10, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='left', va='bottom', fontsize=fontsize1)
autolabel(bars)
#gmmaccax.
gmmaccax.legend()
#gmmaccfig = plt.figure()
#gmmaccax = gmmaccfig.add_subplot(111)
#gmmaccax.bar(styleIndex,accuracyText,tick_label=styleText)
#gmmaccax.set_title("GMM ACCURACIES")
gmmaccfig.tight_layout()
plt.show()







kmeans = KMeans(init='k-means++',n_clusters=numClusters, n_init=20,max_iter=1000,verbose=0)
labels3 = kmeans.fit_predict(reducedData)



clustersBrokenUp2=[] #this just splits the labels array into smaller arrays so I can do mode
for cluster in range(numClusters):
    clustersBrokenUp2.append([])

uniqueLabels=np.unique(paintingImageLabels).tolist()
intLabels2=[]
for labelIndex in range(len(paintingImageLabels)):
    #print("LABELELELEL",paintingImageLabels[labelIndex],str(paintingImageLabels[labelIndex])=='nan',paintingImageLabels[labelIndex]==None)
    intLabels2.append(uniqueLabels.index(paintingImageLabels[labelIndex]))
    clustersBrokenUp2[labels3[labelIndex]].append(uniqueLabels.index(paintingImageLabels[labelIndex]))
#print("kFINAL******\n",intLabels,'\n',labels2,'\n',clustersBrokenUp)


totalInSubClusterArray=0
#sanity check
for subcluster in range(len(clustersBrokenUp2)):
    totalInSubClusterArray+=len(clustersBrokenUp2[subcluster])
print("kSANITTY",totalInSubClusterArray,len(paintingImageLabels))
'''
for cluster in range(numClusters):
    cat2=(labels2==cluster)
    cat=(labels==cluster)
    labels2[cat2]=mode(intLabels[cat2])
    labels[cat]=mode(intLabels[cat])'''
    
'''

figure out the mode genre label of each cluster

'''

desiredLabelForEachCluster2=[]
for cluster in range(numClusters):
    c=Counter(clustersBrokenUp2[cluster])
    desiredLabelForEachCluster2.append(c.most_common(1)[0][0])
print("kDESIRED",desiredLabelForEachCluster2)

c1=Counter(intLabels2)
print("k***********c1\n",c1.keys(),'\n',c1.values(),'\n^^^^^^^^^^^^^^^^^^^^^^^\n')



accuracyOfEachCluster2=[]
accuracyOfDesiredInEachCluster2=[]

labelsCopied2=labels3.copy()

for cluster in range(numClusters):
    howManyOfDesiredLabelShouldBeInThisCluster=intLabels2.count(desiredLabelForEachCluster2[cluster])
    for l in range(len(labelsCopied2)):
        if labelsCopied2[l] == cluster:
            labelsCopied2[l]=desiredLabelForEachCluster2[cluster]
    #howManyOfDesiredLabelAreInThisCluster
    numCorrectInThisCluster=0
    for e in clustersBrokenUp2[cluster]:
        if e == desiredLabelForEachCluster2[cluster]:
            numCorrectInThisCluster+=1
    accuracyOfEachCluster2.append(numCorrectInThisCluster/len(clustersBrokenUp2[cluster]))
    accuracyOfDesiredInEachCluster2.append(numCorrectInThisCluster/howManyOfDesiredLabelShouldBeInThisCluster)
    
# i can also measure how many of the elements in ALL of the labels with the desired label were in that cluster
print("kacc",accuracyOfEachCluster2,'\n\n\n',accuracyOfDesiredInEachCluster2,'\n\n\n',sum(accuracyOfEachCluster2)/len(accuracyOfEachCluster2),
      sum(accuracyOfDesiredInEachCluster2)/len(accuracyOfDesiredInEachCluster2),'\n\n\n%%%%%%%%%%%%%%%%%')


acc3=accuracy_score(intLabels2,labelsCopied2)
print("kaccuracies",acc3)

accuracyForEachStyle2=[]

for style in range(len(allPaintingStyles)):
    howManyOfThisStyleThereAre=countOfEachPaintingStyle[allPaintingStyles[style]]
    howManyOfThisStyleAreInEachCluster=[]
    for cluster in range(numClusters):
        howManyOfThisStyleAreInEachCluster.append(0)
    #print("$",howManyOfThisStyleAreInEachCluster)
    for thisLabel in range(len(labels3)):
        howManyOfThisStyleAreInEachCluster[labels3[thisLabel]]+=1
    
    maxNumberOfThisStyleInAnyCluster=-1
    for cluster in range(numClusters):
        if howManyOfThisStyleAreInEachCluster[cluster]>maxNumberOfThisStyleInAnyCluster:
            maxNumberOfThisStyleInAnyCluster=howManyOfThisStyleAreInEachCluster[cluster]
    #print("----------------accuracy for this style:",allPaintingStyles[style],round(howManyOfThisStyleThereAre/maxNumberOfThisStyleInAnyCluster,4),"n=",howManyOfThisStyleThereAre)
    accuracyForEachStyle2.append((round(howManyOfThisStyleThereAre/maxNumberOfThisStyleInAnyCluster,4),allPaintingStyles[style],howManyOfThisStyleThereAre))
    
sortedVals2=sorted(accuracyForEachStyle2, key=lambda x: x[0],reverse=True)

print('kk(((((((((((((((((((((',sortedVals2)

























'''color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.show()

plot_results(reducedData, gmm.predict(reducedData), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#print("REDUCED DATA",reducedData[:, 2])
ax.scatter(reducedData[:, 0][::visualsSkip], reducedData[:, 1][::visualsSkip], reducedData[:, 2][::visualsSkip],c=labels2[::visualsSkip],marker='.',cmap='gist_rainbow');
ax.set_xlabel('grayscale')
ax.set_ylabel('saturation')
ax.set_zlabel('luminance')
#print(featureMatrix,'\n',featureMatrix[0],'\n',featureMatrix[1],'\n\n\n\n\n',featureMatrix[:, 0])
#for cluster in range(numClusters):
    #print('cluster: ', cluster,np.where(labels == cluster))
    #print(paintingImageLabels[np.where(labels == cluster)])
annotations=[]
textAnnotations=[]


frame_palette = sns.color_palette('Set1',
                                          n_colors=len(np.unique(labels2)))
for i, txt in enumerate(paintingImageLabels[::visualsSkip]):
    #print("feature matrix",featureMatrix[i][0])
    #arr = np.arange(100).reshape((10, 10))
    #ax.annotate(txt, (reducedData[i][0], reducedData[i][1], reducedData[i][2]))
    
    #im.image.axes = ax
    #ax.add_artist(ab)
    
    #print(len(paintingImageLabels),len(reducedData))
    #tag = annotate3D(ax,txt,(reducedData[i][0], reducedData[i][1], reducedData[i][2]), fontsize=10, xytext=(-3,3),
               #textcoords='offset points', ha='right',va='bottom')
    x2, y2, _ = proj3d.proj_transform(reducedData[i*visualsSkip][0],reducedData[i*visualsSkip][1],reducedData[i*visualsSkip][2], ax.get_proj())
    
    #annotationPosition=(random.randrange(0,1)*(1 if random.randint(0,1)>0 else -1),
                        #random.randrange(0,1)*(1 if random.randint(0,1)>0 else -1))
    
    
    
    im = OffsetImage(imagesForOutput[i*visualsSkip], zoom=picZoom)
    im.image.axes = ax
    
    ab = AnnotationBbox(im, (x2,y2),
                        xybox=(0, 0),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.0,
                        arrowprops=dict(arrowstyle="->"),bboxprops = dict(edgecolor=frame_palette[labels2[i*visualsSkip]], linewidth=6))
    ab.set_zorder(-10)
    annotations.append(ab)
    
    ax.add_artist(ab)
    
    #ann=ax.annotate(
        #txt, 
        #xy=(x2, y2), size=5,
        #bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.2))
    ann=Annotation(txt, 
        xy=(x2, y2), size=5,
        bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.9))
    ax.add_artist(ann)
    textAnnotations.append(ann)
    
def update_position1(e):
    for i in range(0,len(paintingImageLabels[::visualsSkip])):
        x2, y2, _ = proj3d.proj_transform(reducedData[i*visualsSkip][0],reducedData[i*visualsSkip][1],reducedData[i*visualsSkip][2], ax.get_proj())
        annotations[i].xy = x2,y2
        annotations[i].update_positions(fig.canvas.renderer)
        
        #textAnnotations[i].xy = x2,y2
        #textAnnotations[i] = x2,y2
        textAnnotations[i].set_position((x2,y2))
        #textAnnotations[i].update_positions(fig.canvas.renderer)
    fig.canvas.draw()
#set_size(100,100,ax)
fig.canvas.mpl_connect('button_release_event', update_position1)
fig.tight_layout()
plt.show()
#plt.ioff()
#plt.close()

#ax2=plt.subplot();
#ax2.scatter(reducedData[:, 0], reducedData[:, 1], c=labels2, s=40, cmap='gist_rainbow');
#set_size(100,100,ax2)
#plt.show()













fig0 = plt.figure()
ax0 = fig0.add_subplot(111, projection='3d')
#print("REDUCED DATA",reducedData[:, 2])
ax0.scatter(reducedData[:, 0][::visualsSkip], reducedData[:, 1][::visualsSkip], reducedData[:, 2][::visualsSkip],c=labels3[::visualsSkip],marker='.',cmap='gist_rainbow');
ax0.set_xlabel('grayscale')
ax0.set_ylabel('saturation')
ax0.set_zlabel('luminance')
#print(featureMatrix,'\n',featureMatrix[0],'\n',featureMatrix[1],'\n\n\n\n\n',featureMatrix[:, 0])
#for cluster in range(numClusters):
    #print('cluster: ', cluster,np.where(labels == cluster))
    #print(paintingImageLabels[np.where(labels == cluster)])
annotations0=[]
textAnnotations0=[]


frame_palette0 = sns.color_palette('Set1',
                                          n_colors=len(np.unique(labels3)))
for i, txt in enumerate(paintingImageLabels[::visualsSkip]):
    #print("feature matrix",featureMatrix[i][0])
    #arr = np.arange(100).reshape((10, 10))
    #ax.annotate(txt, (reducedData[i][0], reducedData[i][1], reducedData[i][2]))
    
    #im.image.axes = ax
    #ax.add_artist(ab)
    
    #print(len(paintingImageLabels),len(reducedData))
    #tag = annotate3D(ax,txt,(reducedData[i][0], reducedData[i][1], reducedData[i][2]), fontsize=10, xytext=(-3,3),
               #textcoords='offset points', ha='right',va='bottom')
    x2, y2, _ = proj3d.proj_transform(reducedData[i*visualsSkip][0],reducedData[i*visualsSkip][1],reducedData[i*visualsSkip][2], ax0.get_proj())
    
    #annotationPosition=(random.randrange(0,1)*(1 if random.randint(0,1)>0 else -1),
                        #random.randrange(0,1)*(1 if random.randint(0,1)>0 else -1))
    
    
    
    im = OffsetImage(imagesForOutput[i*visualsSkip], zoom=picZoom)
    im.image.axes = ax0 
    
    ab = AnnotationBbox(im, (x2,y2),
                        xybox=(0, 0),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.0,
                        arrowprops=dict(arrowstyle="->"),bboxprops = dict(edgecolor=frame_palette0[labels3[i*visualsSkip]], linewidth=6))
    ab.set_zorder(-10)
    annotations0.append(ab)
    
    ax0.add_artist(ab)
    
    #ann=ax.annotate(
        #txt, 
        #xy=(x2, y2), size=5,
        #bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.2))
    ann=Annotation(txt, 
        xy=(x2, y2), size=5,
        bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.9))
    ax0.add_artist(ann)
    textAnnotations0.append(ann)
    
def update_position2(e):
    for i in range(0,len(paintingImageLabels[::visualsSkip])):
        x2, y2, _ = proj3d.proj_transform(reducedData[i*visualsSkip][0],reducedData[i*visualsSkip][1],reducedData[i*visualsSkip][2], ax0.get_proj())
        annotations0[i].xy = x2,y2
        annotations0[i].update_positions(fig.canvas.renderer)
        
        #textAnnotations[i].xy = x2,y2
        #textAnnotations[i] = x2,y2
        textAnnotations0[i].set_position((x2,y2))
        #textAnnotations[i].update_positions(fig.canvas.renderer)
    fig0.canvas.draw()
#set_size(100,100,ax)
fig0.canvas.mpl_connect('button_release_event', update_position2)
fig0.tight_layout()
plt.show()