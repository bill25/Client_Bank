# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 17:13:25 2018

@author: Aouissat_salsabil
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  roc_auc_score, accuracy_score
from fonction_TP3 import tracer_courbe_roc
import matplotlib.pyplot as plt



# d'abord on commence par recuperer les données pour creer notre data frame , on utilisat la methode read_csv
client_bank=pd.read_csv('base_banque.csv',sep=";",encoding = "latin1")

client_bank.describe()
client_bank.shape
# on applique la methode info sur note data frame pour voir un peu sa structure combien de ligne et combien de colonne
#et elle nous permet aussi de trouver les dtypes : object , d'ou on est obliger de les remplacer avec de variable numeric qui referencie a chaque une d'elle
client_bank.info()
#on remarque que on 3 varible des type objet, 
#on commence par la variable I_Salarie_CR 
#on premier temp on applique la fuction unique sue la colonne I_Salarie_CR pour voir les differentes valeur qu'elle peut prende
client_bank['I_Salarie_CR'].unique()
#le valeur sont NS Ret Elp / sont adm(mentienné dans le discriptife)
#(inplace = true) on la mis pour que le changement ce fait au niveau de notre dataframe, pour eviter de faire l'affectation a chaque fois 
client_bank['I_Salarie_CR'].replace({
       'Emp': 0,
       'Ret': 1,
       'Adm': 2,
       'NS' :3}, inplace=True)
#on fait la meme chose avec la varible Nouveau_Client , Metier_portef
client_bank['Nouveau_Client'].unique()
#la variable nouveau client peut prendre 2 variable, on a choisi la valeur 1 pour nouveau client et 0 pour l'ancient client
client_bank['Nouveau_Client'].replace({
        'O' : 1,
        'N' : 0}, inplace=True)

client_bank['Metier_portef'].unique()
#quand on a appliquer la methode unique sur la variable metier prof on a remarquer que la valeur 'parc' dans le discriptif est 'pavs' dans la table
#et elle peut prendre aussi la valeur '0000' en chaine de caracter on va la considerer comme variable maquantre pour le moment et on va la remplacer par nan
client_bank['Metier_portef'].replace({
        'PTBA' : 1,
        'PTPA' : 2,
        'PTPS' : 3,
        'PTAC' : 4,
        'PTAV' : 5,
        'PAVS' : 6,
        'PACS' : 7,
        '0000' : np.nan}, inplace=True)

client_bank.info()
client_bank.dtypes


#on replace les outliers de la colomuns Groupe_GSP par nan afin d'etudier une la propabilité de les supprimer apres
client_bank.Groupe_CSP=client_bank.Groupe_CSP.replace(0,np.nan)
client_bank.Groupe_CSP.unique()
client_bank.Groupe_CSP.isnull().sum()

#on remarque que presque les meme personne ou on a perdu les valeur de l'age leur valeur groupe_gsp manque aussi
client_bank.Age_PP=client_bank.Age_PP.replace(0,np.nan)
client_bank.Age_PP.isnull().sum()

GroupeIsnull=client_bank.loc[client_bank['Groupe_CSP'].isnull(),:]
ageetgroupe=GroupeIsnull.loc[client_bank['Age_PP'].isnull(),:]
colmmunsNull=ageetgroupe.loc[client_bank['Metier_portef'].isnull(),:]


#on remarque que les valeur manquante dans nptre dataframe represente que 3%
#de tout les donnés donc on a choisi de les supprimer


client_bank.isnull().sum()
client_bank=client_bank.dropna()
#
client_bank=client_bank.loc[(client_bank['Age_PP']<80) & (client_bank['Age_PP']>17),:]
new_client=client_bank.dropna()
#on verifier bien que notre nouveau data frame contient aucune valeur null 
new_client.isnull().any().any()




#pour la restriction des columns on utilise la methode de correlation entre les variable 
#chaque fois on trouve une grande correlation entre les variable on choisi une entre elles
#en se basant sur le critaire suivant :
#apres avoir fais quelque recherche sur le web et sur les coditions utiliser par les banque afin d'atribue des pres banquaire
#on a choisi nos variable en se basant sur ces condition et sur leur descriptif du fichier exel
#on commence par crer un data frame qui contient toutes les columns et on enleve la columns Numero_client
filt_df = new_client.loc[:, new_client.columns != 'Numero_Client']
#in afficher la correlation entre les columns avec correlation_matrix
plt.matshow(filt_df.corr())
#on applique la function correlation sur notre data frame en utilisant la valeur absolu des resultat 

c = filt_df.corr().abs()
#on depile le resultat dans une series en utilisant la methode unstack de panda
s = c.unstack()
#et finalment on tie le resultat de notre objet serie
so = s.sort_values(kind="quicksort")
#pour le seuil on a choisi 0.4 
c=0
for i,value in so.iteritems():
    c+=1
    if value>=0.4:
        so=so[c:]
        break
#on enleve aussi les correlation qui sont egal a 1 car elle represent juste une corr entre la meme valeur
c=0
for i,value in so.iteritems():
    c+=1
    if value==1:
      so=so[:c]
      break

#la on creer notre liste qui contient les columns qu'on a jugé d'apres notre recherche
int_to_cat = ['P_P_Auto','I_Salarie_CR','Groupe_CSP','Age_PP','Sit_Fam','N_Adultes_CC','N_enfants_CC','N_Membres_CC','Filtre_exclusion','CA_dom_A','Cap_Accum','Dispo_Mensuel','Flux_Corr','T_P_Habitat','T_Conso','T_P_Familiaux','N_Produits','SM_DAV_A','T_Collecte','T_Liquidités','T_Epargne','T_Livrets']
columns_to_delet=[]
#donc on va parcourir notre objet series et on va stocker les columns qu'on va supprimer apres dans une liste 
#que on applé ici columns_to_delet
for x in int_to_cat:
   for i,value in so.iteritems():
        if x in i :
            i=list(i)
            i.remove(x)
            if ((i[0] in int_to_cat) or (i[0] in columns_to_delet)):
                continue
            else: 
                columns_to_delet=columns_to_delet+i
        else:
            if x in columns_to_delet:
                continue
            else:
                i=list(i)
                if ((i[0] in int_to_cat or i[1] in int_to_cat ) or (i[0] in columns_to_delet or i[1] in columns_to_delet)):
                    continue
                else:
                    i.remove(i[0])
                    columns_to_delet=columns_to_delet+i
#on enleve la redandance des variable pour avoir en fin une liste qui contient tt les variable
columns_to_delet=list(set(columns_to_delet))
#on creer un nouveau data frame qui va etre rafiné (en supprime les columns que on jugé inutile )
dataframe=client_bank
for i in columns_to_delet:
    dataframe = dataframe.drop(i, axis=1)
#on reafiche notre correlation_matrix , et on va voir que elle plus visible
plt.matshow(dataframe.corr())
dataframe.shape #on remarque que notre nouveau dataframe contient 45 columns
dataframe.info()

#on met tt les variable qualitative dans une liste 
columns_categoriel=['Agence','I_Salarie_CR','Groupe_CSP','Nouveau_Client','Sit_Fam','Depart_Resid',
                    'Filtre_exclusion','Ut_Internet','P_PERP','P_Collecte','P_P_Auto','P_P_Etudiant',
                    'P_C_Autre_Carte','P_C_Maestro','P_C_Paiem_DD','P_C_Gold_Visa_DD','P_Ass_Sante',
                    'P_IARD_PJ','P_IARD_Auto']

#on remplace le type de chaque variable par le type category
for elt in columns_categoriel:
    if elt in dataframe.columns:
         dataframe[elt] = dataframe[elt].astype('category')

dataframe.info()
#pour commencer on jette un oeil sur la distrubtion des valeur dans la colmuns PPAuto 
dataframe.P_P_Auto.hist()
#on remarque que on a besoin d'equilibrer nos données d'apprentissage
numero_client=dataframe.Numero_Client
dataframe = dataframe.drop('Numero_Client',axis=1)

train, test = train_test_split(dataframe, test_size=0.3, random_state=1)
X_train_0 = train[train['P_P_Auto']==0]
X_train_1 = train[train['P_P_Auto']==1]

X_train_1 = resample(X_train_1, 
                     replace=True,
                     n_samples=X_train_0.shape[0],
                     random_state=1)

new_train = pd.concat([X_train_0, X_train_1])
# melanger alleatoirement les elements des deux liste
new_train = shuffle(new_train, random_state=1)
# définition du X et du Y sur l'échantillon train
X_train   = new_train.drop('P_P_Auto', axis=1) 
Y_train   = new_train['P_P_Auto']

 # définition du X et du Y sur l'échantillon test
X_test = test.drop('P_P_Auto',axis=1)
Y_test = test['P_P_Auto']
#creation de l'objet logr
logr = LogisticRegression()
#on entreine notre modele sur les x_train equilibré et le y_train
logr.fit(X_train,Y_train)
#on calcule on predis les resultat de x_test
Y_logit = logr.predict(X_test)
#on creer la table crosstab entre les valeur predit et les valeur du test
table = pd.crosstab(Y_test, Y_logit)

print(table)

print('erreur de prédiction :', 1-accuracy_score(Y_test,Y_logit)) 
# erreur de prediction = ~0.1
print('erreur de prédiction sur variable cible :', 
      table[0][1] / (table[0][1] + table[1][1]))
#erreur de prédiction sur variable cible = ~0.18

#pour confirmer le bon fonctionnement de notre modele en utilise ma methode roc_auc_score

probas = logr.predict_proba(X_test)
print('AUC :', roc_auc_score(Y_test, probas[:,1])) 
#le resultat est parfait! AUC = 0.9253528480301576
tracer_courbe_roc(preds=probas, ytest=Y_test)


#Après avoir entrainé notre modèle sur une partie de la base maintenant nous allons  le relancer
#sur l’ensemble des individus qui ont P_P_auto à 0 
#on va commencer par faire quelque changement sur notre data frame on rajoute la colmuns numero_client que on a supprimer precedement
dataframe.info()
dataframe['Numero_Client']=numero_client
#on selectionne tt les client ou la valeur p_p_auto egal a zero
newdata=dataframe.loc[dataframe.P_P_Auto==0,:]
#on sauvgarde les nuero de clients
numero_client=newdata.Numero_Client
#on retir les deux variable une autre fois 
client_True=newdata.drop('P_P_Auto', axis=1)
client_True=client_True.drop('Numero_Client',axis=1) 
#on relance notre modele avec les données des clients d'ou leur variable p_p_auto egal a zero afin de predire le resultat
probas2 = logr.predict_proba(client_True)
#convertion du resultat a une list de probabilité 
probas2=probas2[:,1].tolist()
#on rajoute le resultat a notre dataframe
newdata['probas']=probas2
newdata['Numero_Client']=numero_client
#finalement on ordonne notre data frame selon la columns de la plus grande proba a la plus petite et on prend les premier 10000 clients
newdata=newdata.sort_values(by='probas',ascending=False)
listOfClient = newdata.head(10000)

writer = pd.ExcelWriter('List De Client.xlsx')
listOfClient.to_excel(writer,'listofClient')
writer.save()
