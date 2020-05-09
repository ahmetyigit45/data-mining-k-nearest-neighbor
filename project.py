import csv
import math
import operator
import numpy as np
import scipy
from operator import add
from scipy.spatial import distance


def hiyerarsik_kumeleme(filename):
    data = dict()
    trainingSet = []
    indexes = []
    ''' eğitim için kullanıcak verinin vectöre atanması '''
    with open(filename, 'r') as csvfile:
        indata = csv.reader(csvfile, delimiter=',', quotechar='|')
        p = 0;
        for row in indata:
            indexes.append(p)
            data[row[0]] = [float(x) for x in row[0:]]
            trainingSet.append(data[row[0]])
            p = p + 1;

    '''Uzaklık Matrisinin hesaplanması'''
    while(True):
        print("\nManhattan için 1")
        print("Euclidean için 2")
        print("Minkowski için 3")
        p=input("P değeri Gir: ")
        if int(p) == 1:
            distance_matrix = scipy.spatial.distance.cdist(trainingSet, trainingSet, metric='cityblock', p=1)
            print("Manhattan Uzaklığı Seçildi")
            break
        elif int(p) == 2:
            distance_matrix = scipy.spatial.distance.cdist(trainingSet, trainingSet, metric='euclidean', p=2)
            print("Euclidean Uzaklığı Seçildi")
            break
        elif int(p) == 3:
            distance_matrix = scipy.spatial.distance.cdist(trainingSet, trainingSet, metric='minkowski', p=3)
            print("Minkowski Uzaklığı Seçildi")
            break
    ''' verileri Kare matris olarak dönüştürür'''
    row, col = distance_matrix.shape

    def mindist(distance_matrix):
        row, col = distance_matrix.shape
        iu1 = np.triu_indices(row, 1)
        return np.where(distance_matrix == np.min(distance_matrix[iu1]))

    '''len_mat, artan küme numarasının kaydını tutmak için kullanılır,
      aralarındaki mesafeyle birlikte bir noktada birleştirilen iki kümenin saklanması için temp kullanılır. 
    '''
    len_mat = row
    tempMatrix = []
    label_list = np.array([x for x in range(row)])

    '''cluster_index, hangi kümelerin birleştirildiğini depolamak için kullanılan bir matristir.,
     yani tekli kümelerle başlar ve sonunda oluşan son kümelerde dizinler içerir'''

    clusters = dict(zip(indexes, trainingSet))
    cluster_index = []
    for i in range(len_mat):
        cluster_index.append([i])

    cluster_id = input("Kümeleme (K değeri) Sayısını Giriniz => ")
    k=int(cluster_id)
    while (distance_matrix.shape != (k, k)):
        tuple_max, tuple_2 = mindist(distance_matrix)
        val1 = tuple_max[0]
        val2 = tuple_max[1]
        d_max_value = distance_matrix[val1][val2]
        distance_matrix = np.array(distance_matrix)

        v = cluster_index[val2]
        '''kümenin oluşturulduğu veri dizini küme dizinine birbirine eklenir'''
        cluster_index.remove(v)
        x = 0
        for i in v:
            x = i
            cluster_index[val1].append(x)

        '''birleştirme kümeleri ve aralarındaki mesafe bağlantı tempMatrix içinde saklanır'''
        tempMatrix.append([label_list[val1], label_list[val2], d_max_value])
        label_list = np.delete(label_list, val2)
        label_list[val1] = len_mat  # yeni kümeleme listesi oluşturuyor
        len_mat += 1  # matrisin boyutunu 1'er 1'er arttırıyor

        '''     minimum mesafe kuralına göre hesaplama yapar.   '''
        n = np.minimum(distance_matrix[val1], distance_matrix[val2])

        '''mesafe matrisinden eski değerleri silme ve
         yukarıda kullanılan yönteme bağlı olarak yenilerini ekleme'''
        val = distance_matrix[0, 1]
        new_mat = np.delete(n, val2, 0)
        distance_matrix = np.delete(distance_matrix, val2, 0)
        distance_matrix = np.delete(distance_matrix, val2, 1)
        distance_matrix[val1] = new_mat
        distance_matrix[:, val1] = new_mat
        distance_matrix[val1, val1] = 0
        if (distance_matrix.shape == (1, 1)):
            distance_matrix[val1, val1] = val

    return (tempMatrix, distance_matrix, clusters, cluster_index)

'''  eğitim veri setini hiyerarşik kümeleme methoduye gönderip sonucunda;
uzaklık matrisi, kümeleme metrisi, k değerine göre kümelenen verileri ve temp matrisini oluşturur '''
temp_matrix,distance_matrix,clusters,cluster_index =hiyerarsik_kumeleme('data/train.csv')

''' her kümede birleştirilen veri noktalarının dizinlerini yazdırır     '''
print("\n cluster_index \n")
print('\n'.join(map(str,cluster_index)))

''' matris terimini küme numaraları ve aralarındaki mesafe ile yazdırır    '''
print("\n Temp matrix \n")
print('\n'.join(map(str,temp_matrix)))

''' belirlenen uzaklık türüne göre uzaklık matris oluşturur '''
print("\n Uzaklık Martisi \n")
print('\n'.join(map(str,distance_matrix)))

final = dict()
m = len(cluster_index)

''' kümeleme sayısına göre kümelenen matrisleri gösterir    '''
for i in range(m):
    count = 0;
    for lists in cluster_index[i]:
        if count == 0:
            final[i] = clusters[i]
            count += 1
        elif count > 0:
            final[i] = map(add, final[i], clusters[lists])
            count += 1
    final[i] = [x / count for x in final[i]]
print("\nK değerine göre kümelenen matrisler")
for x in final:
    print("\nKüme Sayısı:  " + str(x+1))
    print(final[x])

'''     eğitim verileri sonucunda oluşan sonuçlar, test verileri ile  knn ile analiz edilecek     '''
def knn(filename):
    with open(filename, 'r') as csvfile:
        d = dict()
        '''  test verilerini  testingSet dizisine atıyor  '''
        testingSet = []
        testdata = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in testdata:
            d[row[0]] = [float(x) for x in row[0:]]
            testingSet.append(d[row[0]])
    resultmatrix = dict()

    ''' eğitim verilerine göre hesaplanan küme verileri ile test verileri ile kıyaslama yapılır '''
    while (True):
        print("\nEğitilen Modele test verileri verilmek üzere")
        print("Manhattan için 1")
        print("Euclidean için 2")
        print("Minkowski için 3")
        p = input(" 1 ile 3 P Değeri Giriniz => ")
        if int(p) == 1 or int(p) == 2 or int(p) == 3:
            if int(p) == 1:
                print("Manhattan Uzaklık Hesaplaması Seçildi")
                for k in range(len(testingSet)):
                    cluster = neighbors(final, testingSet[k], int(p))
                    resultmatrix[k] = cluster
                return resultmatrix
            elif int(p) == 2:
                print("Euclidean Uzaklık Hesaplaması Seçildi")
                for k in range(len(testingSet)):
                    cluster = neighbors(final, testingSet[k], int(p))
                    resultmatrix[k] = cluster
                return resultmatrix
            else:
                print("Minkowski Uzaklık Hesaplaması Seçildi")
                for k in range(len(testingSet)):
                    cluster = neighbors(final, testingSet[k], int(p))
                    resultmatrix[k] = cluster
                return resultmatrix
        elif int(p) == 0:
            break
        else:
            print("Geçersiz P değeri girildi")

''' manhattan, öklid ve minkowski uzaklık methodları '''

''' Manhattan distance calcualtion p=1  '''
def manhattanDistance(instance1, instance2, length):
	p=1
	man_Distance = 0
	for i in range(length):
		man_Distance += (abs(instance1[i] - instance2[i]))
	return man_Distance

''' euclidean distance calcualtion p=2  '''
def euclideanDistance(instance1, instance2, length):
	p=2
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

'''  Minkowski distance calcualtion p=3 '''
def minkowskiDistance(instance1, instance2, length):
	p=3
	minkowski_distance = 0
	for x in range(length):
		minkowski_distance += pow(abs(instance1[x]-instance2[x]),p)
		minkowski_distance = pow(minkowski_distance, (1/p))
	return minkowski_distance

def neighbors(trainingSet, testInstance,p):
    distances = []
    length = len(testInstance)
    for x in range(len(trainingSet)):
        if p == 1:
            dist = manhattanDistance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        elif p == 2:
            dist = euclideanDistance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        else:
            dist = minkowskiDistance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    return (distances[0])

result_matrix = knn("data/test.csv")
for x in result_matrix:
    print("\n" + str(x) + ". Test verisi ile  " + str(result_matrix[x][0]) + ". küme ile arasındaki uzaklık: " + str(result_matrix[x][1]))