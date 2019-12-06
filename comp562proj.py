
print("test")
srcPath='images_5_classes'
import numpy as np
import gzip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
        
import pandas

paintingMetadata=pandas.read_csv("___paintingdata/all_data_info.csv")
print("**********************painting metadata",'\n\n',paintingMetadata,'\n\n',paintingMetadata['new_filename'],'\n\n',
      paintingMetadata.columns,'\n\n',paintingMetadata.columns[0],'\n\n',
      paintingMetadata.loc[paintingMetadata['new_filename'] == '1.jpg'].to_string(),'\n\n',
      paintingMetadata.loc[paintingMetadata['new_filename'] == '1.jpg']['artist'].to_string(),'\n\n',
      (paintingMetadata.loc[paintingMetadata['new_filename'] == '1.jpg']['artist']=='M.C. Escher').to_string(),'\n *************************************************',
      len(paintingMetadata.loc[paintingMetadata['style'] == 'Realism']))

from os import listdir
from os.path import isfile, join
import collections
import time
import sklearn.mixture
from numpy import array

paintingPicturesPath='___paintingdata/train_1'
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
print("|||||||||||||||||||||||||||||||||||",paintingImageFeatures,'\npppppppppppppppppppp',test3)
paintingImageLabels=[]
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
skips=[3,3]
showImages=False
i=0

rangeOfImagesToProcess=3000
rangeOfImagesToShow=rangeOfImagesToProcess
start_time = time.time()
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",type(4) is int)
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
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^not an image?",painting)
    else:
        paintingImageLabels.append(imageStyle)
        averageRGB=[0,0,0]
        averageColorGrayscale=0;
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
                if thisPixelGrayscale<lightestColorInGrayscale:
                    lightestColorInGrayscale=round(thisPixelGrayscale,4)
                    lightestColorInRGB=[round(thisPixelRGB[0],4),round(thisPixelRGB[1],4),round(thisPixelRGB[2],4)]
                if thisPixelGrayscale>darkestColorInGrayscale:
                    darkestColorInGrayscale=round(thisPixelGrayscale,4)
                    darkestColorInRGB=[round(thisPixelRGB[0],4),round(thisPixelRGB[1],4),round(thisPixelRGB[2],4)]
                if thisPixelGrayscale>=0.5:
                    averageDarkness+=thisPixelGrayscale
                    numDarkPixels+=1
                if thisPixelGrayscale<0.5:
                    averageBrightness+=thisPixelGrayscale
                    numBrightPixels+=1
                #print("this pixel rgb",thisPixelRGB,averageRGB)#,thisPainting[thisPixelX][thisPixelY])
        averageRGB=[round(averageRGB[0]/numPixels,4),round(averageRGB[1]/numPixels,4),round(averageRGB[2]/numPixels,4)]
        averageColorGrayscale=round(averageColorGrayscale/numPixels,4)
        
        
        averageBrightness=round(averageBrightness/numBrightPixels,4) if numBrightPixels>0 else 0.5
        averageDarkness=round(averageDarkness/numDarkPixels,4) if numDarkPixels>0 else 0.5
        
        paintingImageFeatures.append([averageRGB[0],averageRGB[1],averageRGB[2],
                                      averageColorGrayscale,
                                      darkestColorInRGB[0],darkestColorInRGB[1],darkestColorInRGB[2],
                                      lightestColorInRGB[0],lightestColorInRGB[1],lightestColorInRGB[2],
                                      darkestColorInGrayscale,
                                      lightestColorInGrayscale,
                                      round(darkestColorInGrayscale-lightestColorInGrayscale,4),
                                      averageDarkness,
                                      averageBrightness,
                                      round(averageDarkness-averageBrightness,4)])
        #print("333333333333333333333333",paintingImageFeatures)
        i+=1
        #print("image features",i,paintingImageFeatures[len(paintingImageFeatures)-1], paintingImageLabels[len(paintingImageFeatures)-1])

print(paintingImageLabels.count('Baroque'),'\n processed in ',time.time() - start_time,'\n',collections.Counter(paintingImageLabels))
#print("{{{{{{{{____",paintingImageFeatures)
featureMatrix=np.stack(paintingImageFeatures)
#print("[[[[[[[[[[[[[[[[____",featureMatrix)
#featureMatrix=np.asmatrix(paintingImageFeatures)
#print(featureMatrix)
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
class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)
        
        

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)
    
print("this many features=",len(paintingImageFeatures[0]),'___this many labels',len(collections.Counter(paintingImageLabels)))
#print(gmm)
numClusters=len(collections.Counter(paintingImageLabels))
numClusters=3
from sklearn import preprocessing
#reducedData=PCA(n_components=numClusters).fit_transform(featureMatrix)
reducedData=preprocessing.StandardScaler().fit_transform(PCA(n_components=numClusters).fit_transform(featureMatrix))
#reducedData=pandas.DataFrame(reducedData,columns=featureMatrix.columns)
kmeans = KMeans(init='k-means++',n_clusters=numClusters, n_init=10)
labels = kmeans.fit_predict(reducedData)


gmm=sklearn.mixture.GaussianMixture(n_components=numClusters,covariance_type='full').fit(reducedData)
labels2=gmm.predict(reducedData)
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn import datasets


uniqueLabels=np.unique(paintingImageLabels).tolist()
intLabels=[]
for label in paintingImageLabels:
    print(label,str(label)=='nan',label==None)
    intLabels.append(uniqueLabels.index(label))
print(intLabels,labels2)
'''for i in range(numClusters):
    cat2=(labels2==i)
    cat=(labels==i)
    labels2[cat2]=mode(intLabels[cat2])
    labels[cat]=mode(intLabels[cat])'''

acc=accuracy_score(intLabels,labels)
acc2=accuracy_score(intLabels,labels2)
print("accuracies",acc,acc2)
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print("REDUCED DATA",reducedData[:, 2])
ax.scatter(reducedData[:, 0], reducedData[:, 1], reducedData[:, 2],c=labels2,marker='o',cmap='gist_rainbow');
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
#print(featureMatrix,'\n',featureMatrix[0],'\n',featureMatrix[1],'\n\n\n\n\n',featureMatrix[:, 0])
#for cluster in range(numClusters):
    #print('cluster: ', cluster,np.where(labels == cluster))
    #print(paintingImageLabels[np.where(labels == cluster)])
annotations=[]
textAnnotations=[]

import random
import seaborn as sns
frame_palette = sns.color_palette('Set1',
                                          n_colors=len(np.unique(labels2)))
for i, txt in enumerate(paintingImageLabels):
    #print("feature matrix",featureMatrix[i][0])
    #arr = np.arange(100).reshape((10, 10))
    #ax.annotate(txt, (reducedData[i][0], reducedData[i][1], reducedData[i][2]))
    
    #im.image.axes = ax
    #ax.add_artist(ab)
    
    print(len(paintingImageLabels),len(reducedData))
    #tag = annotate3D(ax,txt,(reducedData[i][0], reducedData[i][1], reducedData[i][2]), fontsize=10, xytext=(-3,3),
               #textcoords='offset points', ha='right',va='bottom')
    x2, y2, _ = proj3d.proj_transform(reducedData[i][0],reducedData[i][1],reducedData[i][2], ax.get_proj())
    
    #annotationPosition=(random.randrange(0,1)*(1 if random.randint(0,1)>0 else -1),
                        #random.randrange(0,1)*(1 if random.randint(0,1)>0 else -1))
    
    
    
    im = OffsetImage(mpimg.imread(paintingPicturesPath+'/'+paintingPictures[i]), zoom=0.1)
    im.image.axes = ax
    
    ab = AnnotationBbox(im, (x2,y2),
                        xybox=(0, 0),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.0,
                        arrowprops=dict(arrowstyle="->"),bboxprops = dict(edgecolor=frame_palette[labels2[i]], linewidth=6))
    ab.set_zorder(-10)
    annotations.append(ab)
    
    ax.add_artist(ab)
    
    #ann=ax.annotate(
        #txt, 
        #xy=(x2, y2), size=5,
        #bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.2))
    ann=Annotation(txt, 
        xy=(x2, y2), size=5,
        bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.2))
    ax.add_artist(ann)
    textAnnotations.append(ann)
    
def update_position(e):
    for i in range(0,len(paintingImageLabels)):
        x2, y2, _ = proj3d.proj_transform(reducedData[i][0],reducedData[i][1],reducedData[i][2], ax.get_proj())
        annotations[i].xy = x2,y2
        annotations[i].update_positions(fig.canvas.renderer)
        
        #textAnnotations[i].xy = x2,y2
        #textAnnotations[i] = x2,y2
        textAnnotations[i].set_position((x2,y2))
        #textAnnotations[i].update_positions(fig.canvas.renderer)
    fig.canvas.draw()
#set_size(100,100,ax)
fig.canvas.mpl_connect('button_release_event', update_position)
#plt.show()
plt.ioff()
plt.close()
'''
#ax2=plt.subplot();
#ax2.scatter(reducedData[:, 0], reducedData[:, 1], c=labels2, s=40, cmap='gist_rainbow');
#set_size(100,100,ax2)
#plt.show()