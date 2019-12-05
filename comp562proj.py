
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
plt.hist(x, bins=[0.5,1.5,2.5,3.5,4.5,5.5], align='mid', edgecolor='black')
plt.xlabel('X')
plt.ylabel('count')
plt.show()
mus =    [1.0,10.0,-7.0]
sigmas = [1.0, 1.0, 2.0]
for (mu,sigma) in zip(mus,sigmas):
    x = np.random.normal(mu,sigma,2000)
    print ("Mean: " + str(np.mean(x)) + " Standard Deviation: " + str(np.std(x)))
    plt.xlim(-15,15)
    plt.hist(x,40,normed=True,label='mu:'+str(mu)+' sigma:'+str(sigma))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()
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
        #plt.set_xticklabels([])
        #plt.set_yticklabels([])
        #plt.set_xticks([])
        #plt.set_yticks([])
        plt.xticks()
        plt.yticks()
        plt.xlabel("")
        plt.ylabel("")
        plt.subplots_adjust(wspace=0, hspace=0)
        print("showing",(srcPath + '/' + folderName.decode('utf-8') + '/' + fID.decode('utf-8') + '.jpg'))
        plt.imshow(img)
        #plt.axis('off') 
        plt.show()
#plt.show()
