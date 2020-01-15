import os
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import numpy as np

def getHMName(fname):
    hmName = ""
    index = 0
    if fname.upper().__contains__("H2"):
        index = fname.upper().index("H2")
    elif fname.upper().__contains__("H3"):
        index = fname.upper().index("H3")
    elif fname.upper().__contains__("H4"):
        index = fname.upper().index("H4")
    else:
        print("ERROR!!")
        print(fname)
        return "Extra"

    if fname.upper().__contains__("AC"):
        indexLast = fname.upper().index("AC")
        hmName = fname[index:indexLast + 2]
        hmName = hmName.split("_")[0]
    elif fname.upper().__contains__("ME"):
        indexLast = fname.upper().index("ME")
        hmName = fname[index:indexLast + 3]
        hmName = hmName.split("_")[0]
    elif fname.upper().__contains__("TETRA"):
        indexLast = fname.upper().index("TETRA")
        hmName = fname[index:indexLast + 5]
        hmName = hmName.split("_")[0]
        #print(fname)
    else:
        hmName = fname[index:index + 2]
        hmName = hmName.split("_")[0]
    #print(hmName)
    hmName = hmName.split(":")[0]
    return hmName
def txt2cvs(input,output):
    ifstream = open(input, "r")
    ofstream = open(output, "w")

    line = ifstream.readline().strip()
    while line!= "":
        lparts = line.split()
        #tempstr = ""
        #for p in lparts:
        #    tempstr += p +","
        ofstream.write(lparts[0] + "," + lparts[2] + "\n")
        line = ifstream.readline().strip()

    ifstream.close()
    ofstream.close()
def txt2cvsDom(input,output, hmfile):
    ifstream = open(input, "r")
    ofstream = open(output, "w")
    hmstream = open(hmfile, "r")
    hmList = []
    templine = hmstream.readline()
    templine = hmstream.readline().strip()
    while templine != "":
        hmList.append(templine)
        templine = hmstream.readline().strip()
    hmstream.close()

    hmstr = ','.join(map(str, hmList))
    line = ifstream.readline()
    line = ifstream.readline().strip().split()
    ofstream.write(line[0] + "," + line[1] + "," + hmstr + "\n")
    line = ifstream.readline().strip()
    while line!= "":
        lparts = line.split()
        templist = [0] * hmList.__len__()
        if int(lparts[1])>0:
            hms = lparts[2].split(",")
            if hms[hms.__len__()-1] == "":
                hms.pop()
            for temph in hms:
                hm = temph.split(":")[0]
                ind = hmList.index(hm)
                templist[ind] = temph.split(":")[1]
        tempstr = ','.join(map(str, templist))
        ofstream.write(lparts[0] + "," + lparts[1] + "," + tempstr + "\n")
        line = ifstream.readline().strip()

    ifstream.close()
    ofstream.close()
def ClusterSignalFiles():
    stages = ["Emb", "L3"]
    stages = ["Emb"]
    chrms = ["I"]
    Directory = "/home/boudela/mountboudela/HisMod/"
    for stage in stages:
        fileDir = Directory + stage + "/"
        folders = os.listdir(fileDir)
        for chrm in chrms:
            for folder in folders:
                if folder.startswith("modEncode") and folder.startswith("modEncode_2726"):
                    print(folder)
                    newDir = fileDir + folder + "/signal_data_files/"
                    files = os.listdir(newDir)
                    for file in files:
                        if file.endswith(".wig"):
                            txtF = newDir+file.split(".")[0] + "_" + chrm + ".txt"
                            csvF = newDir+file.split(".")[0] + "_" + chrm + ".csv"
                            txt2cvs(txtF, csvF)
                            df = pd.read_csv(csvF, sep=',')
                            #print(df)
                            df_tr = df
                            kmeans = KMeans(n_clusters=4)
                            kmeans.fit(df)
                            labels = kmeans.predict(df)
                            centroids = kmeans.cluster_centers_
                            plt.scatter( df['peakValue'],df['sp'], c=kmeans.labels_)
                            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, color="red")  # Show the centres
                            plt.show()
                            '''
                            fig = plt.figure(figsize=(5, 5))
                            colmap = {1: 'r', 2: 'g', 3: 'b'}
                            colors = map(lambda x: colmap[df['peakValue'] + 1], labels)

                            plt.scatter(df['sp'], df['peakValue'], color=df['color'], alpha=0.5, edgecolor='k')
                            for idx, centroid in enumerate(centroids):
                                plt.scatter(*centroid)
                            plt.xlim(0, 80)
                            plt.ylim(0, 80)
                            plt.show()

                             
                            wcss = []
                            for i in range(1, 11):
                                kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                                kmeans.fit(df_tr)
                                wcss.append(kmeans.inertia_)
                                print(i)
                            plt.plot(range(1, 11), wcss)
                            plt.title('Elbow Method')
                            plt.xlabel('Number of clusters')
                            plt.ylabel('WCSS')
                            plt.show()
                            # Platou observed at 4, thus 4 clusters
                            '''




                            clmns = ['sp', 'peakValue']
                            #plt.scatter(df_tr['sp'][:100], df_tr['peakValue'][:100])
                            #plt.show()


                            '''
                            kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
                            pred_y = kmeans.fit_predict(df_tr)
                            plt.scatter(df_tr['sp'][:10], df_tr['peakValue'][:10])
                            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
                            plt.show()

                            
                            #df_tr_std = stats.zscore(df_tr['peakValue'])
                            # Cluster the data (High sig, low sig, noise)
                            kmeans = KMeans(n_clusters=3, random_state=0).fit(df_tr)
                            labels = kmeans.labels_

                            # Glue back to originaal data
                            df_tr['clusters'] = labels

                            # Add the column into our list
                            clmns.extend(['clusters'])

                            # Lets analyze the clusters
                            print(df_tr[clmns].groupby(['clusters']).mean())
                            
                            
                            
                            kmeans = KMeans(n_clusters=3)
                            kmeans.fit(df)
                            labels = kmeans.predict(df)
                            centroids = kmeans.cluster_centers_
                            fig = plt.figure(figsize=(5, 5))

                            colors = map(lambda x: colmap[x + 1], labels)

                            plt.scatter(df['sp'], df['peakValue'], color=colors, alpha=0.5, edgecolor='k')
                            for idx, centroid in enumerate(centroids):
                                plt.scatter(*centroid, color=colmap[idx + 1])
                            plt.xlim(0, 80)
                            plt.ylim(0, 80)
                            plt.show()
                            
                            df = pd.DataFrame({
                                'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
                                'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
                            })
                            
                            np.random.seed(200)
                            k = 3
                            # centroids[i] = [x, y]
                            centroids = {
                                i + 1: [np.random.randint(0, 80), np.random.randint(0, 80)]
                                for i in range(k)
                            }

                            fig = plt.figure(figsize=(5, 5))
                            plt.scatter(df['x'], df['y'], color='k')
                            colmap = {1: 'r', 2: 'g', 3: 'b'}
                            for i in centroids.keys():
                                plt.scatter(*centroids[i], color=colmap[i])
                            plt.xlim(0, 80)
                            plt.ylim(0, 80)
                            plt.show()
                            
                            
                            
                            df_tr = df
                            
                            clmns = ['peakValue']
                            df_tr_std = stats.zscore(df_tr['peakValue'])
                            print(df_tr_std.describe())
                            #Cluster the data (High sig, low sig, noise)
                            kmeans = KMeans(n_clusters=3, random_state=0).fit(df_tr_std)
                            labels = kmeans.labels_

                            #Glue back to originaal data
                            df_tr['clusters'] = labels

                            #Add the column into our list
                            clmns.extend(['clusters'])

                            #Lets analyze the clusters
                            print (df_tr[clmns].groupby(['clusters']).mean())
                            '''

def ClusterDomainFiles():

    stages = ["Emb", "L3"]
    #stages = [ "L3"]
    chrms = ["I"]

    for stage in stages:
        print(stage)
        txtF = "/home/boudela/mountboudela/HisMod/" + stage + "/HMfiles/listOfHMpeaks.txt"
        csvF = txtF.split(".")[0] + ".csv"
        hmfile =  "/home/boudela/mountboudela/HisMod/" + stage + "/HMfiles/listOfHmNames.txt"
        #txt2cvsDom(txtF, csvF, hmfile)

        df = pd.read_csv(csvF, sep=',')
        df_orig = df
        df = df.drop(columns=['Window'])
        df = df.drop(columns=['HMCount'])

        print(df)
        df_tr = df
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(df)
        labels = kmeans.predict(df)
        print("+++++++++++++++")
        print(labels)
        print("+++++++++++++++")
        '''
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(df_tr)
            wcss.append(kmeans.inertia_)
            print(i)
        plt.plot(range(1, 11), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        # Platou observed at 5, thus 4 clusters
        '''
        centroids = kmeans.cluster_centers_
        plt.scatter(df_orig['Window'], labels, c=kmeans.labels_)
        #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, color="red")  # Show the centres
        plt.xlabel('Window')
        plt.ylabel("labels")
        plt.title(stage)
        plt.show()


ClusterSignalFiles()
ClusterDomainFiles()