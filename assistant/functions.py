import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score

classes = 'table table-sm table-striped table-hover'
nulls = ['Null','NaN','',' ','#N/A','#N/D','?','#','nan']

def handle_uploaded_file(f):  
    with open('media/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():  
            destination.write(chunk)
    return destination.name

def delete_file(path):
   """ Deletes file from filesystem. """
   if os.path.isfile(path):
       os.remove(path)

def step_1(request, file=None, df = None):


    if df is None:
        df = pd.read_csv(file, na_values=nulls)
        df = df.replace(nulls,np.nan)
        df.columns = [i.title().strip().replace(' ','_') for i in df.columns.to_list()]

        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except:
                pass

        df.to_pickle('media/{}_step1.pkl'.format(request.COOKIES['csrftoken']))
        delete_file(file)        
    
    sample = df.sample(10)
    sample = sample.to_html(classes=classes, justify='center',border=0, index=False)

    n_rows = df.shape[0]
    n_cols = df.shape[1]
    cols = df.columns.to_list()

    data_frame = {
        'sample':sample,
        'n_rows':n_rows,
        'n_cols':n_cols,
        'cols':cols,
    }

    return data_frame

    # return load_pickle(request, 1)

def step_2(request): 
    df = load_pickle(request, 1)
    cols = df.columns
    info = []
    for c in cols:
        try:
            df[c] = pd.to_numeric(df[c])
        except:
            pass

    for c in cols:
        info.append(
            {
                "column":c, 
                "null":df[c].isnull().sum(), 
                "notnull":df[c].notnull().sum(), 
                "dtype":str(df.dtypes[c])
            }
        )

    describe = df.describe(include="all")
    describe = describe.to_html(classes=classes, justify='center',border=0, )

    plt.figure(figsize=(15,10))
    sb.boxplot(data=pd.DataFrame(df,columns=df.columns))
    plt.xticks(rotation=90)
    plt.savefig('media/{}_boxplot.jpg'.format(request.COOKIES['csrftoken']))

    vbs_categoricas = df.columns.tolist()
    vbs_cortas =[]
    vbs_largas =[]

    for v in vbs_categoricas:
        if len(df[v].unique())<=3:
            vbs_cortas.append(v)
        else:
            vbs_largas.append(v)

    if len(vbs_cortas) == 1:
        plt.figure(figsize=(5,5))
        sb.boxplot(data=pd.DataFrame(vbs_cortas[0],label='Count'))
        plt.savefig('media/{}_vbs_cortas.jpg'.format(request.COOKIES['csrftoken']))
    else:
        plt.figure(figsize=(30,30))
        for idx, col in enumerate(vbs_cortas,1):
            plt.subplot(math.ceil(len(vbs_cortas)/4),math.ceil(len(vbs_cortas)/4),idx)
            sb.countplot(x=df[col],label='Count')
        plt.savefig('media/{}_vbs_cortas.jpg'.format(request.COOKIES['csrftoken']))

    plt.figure(figsize=(30,30))
    sb.pairplot(df[vbs_largas], corner=True ,diag_kind="hist",palette="bright")
    plt.savefig('media/{}_vbs_largas.jpg'.format(request.COOKIES['csrftoken']))



    return {
        "info": info,
        "describe":describe,
        "boxplot":'media/{}_boxplot.jpg'.format(request.COOKIES['csrftoken']),
        "vbs_cortas":'media/{}_vbs_cortas.jpg'.format(request.COOKIES['csrftoken']),
        "rectificar":'media/{}_vbs_largas.jpg'.format(request.COOKIES['csrftoken']),
            
    }

def step_3(request):
    df = load_pickle(request, 1)
    cols = df.columns.tolist()
    return cols

def step_4(request):
    df = load_pickle(request, 1)
    cols = df.columns.tolist()
    target = cols[int(request.GET.get('target'))]
    cols.remove(target)

    unique = {}
    median = {}
    mean = {}
    
    for c in cols:
        try:
            coldata = df[df[c].notnull()][c]
            unique[c] = sorted(coldata.unique())
        except Exception as e:
            pass

    for c in cols:
        try:
            mean[c] = round(df[c].mean(),3)
            median[c] = round(df[c].median(),3)
        except Exception as e:
            mean[c] = 0
            median[c] = 0
            pass

    return {
        "target_id": request.GET.get('target'),
        "target": target,
        "unique": unique,
        "mean":mean,
        "median":median
    }

def step_5(request, df = None):
    if df is None:
        
        df = load_pickle(request, 1)
        dic_imp = {}
        dic_out = {}

        for r in request.POST.keys():
            try:
                if r.find("out_") >=0 and request.POST.getlist(r):
                    dic_out.setdefault(r.split("out_")[1],pd.Series(request.POST.getlist(r), dtype=df[r.split("out_")[1]].dtype).tolist())
            except Exception as e:
                print(e)
                pass
        
        
        for r in request.POST.keys():
            try:
                if r.find("imp_") >=0 and request.POST.get(r):
                    dic_imp.setdefault(r.split("imp_")[1], np.array(float(request.POST.get(r)),dtype=df[r.split("imp_")[1]].dtype))
            except Exception as e:
                print(e)
                pass
        print(dic_imp)
        df = df[~ df.isin(dic_out)]
        df = df.fillna(dic_imp)
        df = df.dropna()

        df.to_pickle('media/{}_step5.pkl'.format(request.COOKIES['csrftoken']))

    describe = df.describe(include="all")
    describe = describe.to_html(classes=classes, justify='center',border=0, )

    cols = df.columns
    info = []
    for c in cols:
        info.append(
            {
                "column":c, 
                "null":df[c].isnull().sum(), 
                "notnull":df[c].notnull().sum(), 
                "dtype":str(df.dtypes[c])
            }
        )

    plt.figure(figsize=(15,10))
    sb.boxplot(data=pd.DataFrame(df,columns=df.columns))
    plt.xticks(rotation=90)
    plt.savefig('media/{}_boxplot_final.jpg'.format(request.COOKIES['csrftoken']))

    vbs_categoricas = df.columns.tolist()
    vbs_cortas =[]
    vbs_largas =[]

    for v in vbs_categoricas:
        if len(df[v].unique())<=3:
            vbs_cortas.append(v)
        else:
            vbs_largas.append(v)

    vbs_cortas = np.unique(vbs_cortas).tolist()
    vbs_largas = np.unique(vbs_largas).tolist()

    fig = plt.figure(figsize=(35,35))

    for idx, col in enumerate(vbs_cortas,1):
        ax = fig.add_subplot(plt.subplot(math.ceil(len(vbs_cortas)/4),math.ceil(len(vbs_cortas)/4),idx))
        df_pie = df.groupby([col]).size().reset_index(name="Count")
        plt.pie(df_pie['Count'], labels=df_pie[col], autopct='%1.1f%%')
        plt.title(col)

    plt.savefig('media/{}_pie_final.jpg'.format(request.COOKIES['csrftoken']))

    if len(vbs_cortas) == 1:
        plt.figure(figsize=(5,5))
        sb.boxplot(data=pd.DataFrame(vbs_cortas[0],label='Count'))
        plt.savefig('media/{}_vbs_cortas_final.jpg'.format(request.COOKIES['csrftoken']))
    else:
        plt.figure(figsize=(30,30))
        for idx, col in enumerate(vbs_cortas,1):
            plt.subplot(math.ceil(len(vbs_cortas)/4),math.ceil(len(vbs_cortas)/4),idx)
            sb.countplot(x=df[col],label='Count')
        plt.savefig('media/{}_vbs_cortas_final.jpg'.format(request.COOKIES['csrftoken']))


    plt.figure(figsize=(15,10))
    sb.heatmap(df.corr(),cmap="coolwarm",annot=True)
    plt.savefig('media/{}_heatmap_final.jpg'.format(request.COOKIES['csrftoken']))

    cols = df.columns.tolist()
    cols.remove(request.POST.get("target") if request.method == 'POST' else request.GET.get("target"))

    return {
        "describe":describe,
        "target_id": request.POST.get("target_id") if request.method == 'POST' else request.GET.get("target_id"),
        "target": request.POST.get("target") if request.method == 'POST' else request.GET.get("target"),
        "info":info,
        "boxplot":'media/{}_boxplot_final.jpg'.format(request.COOKIES['csrftoken']),
        "pie":'media/{}_pie_final.jpg'.format(request.COOKIES['csrftoken']),
        "vbs_cortas":'media/{}_vbs_cortas_final.jpg'.format(request.COOKIES['csrftoken']),
        "heatmap":'media/{}_heatmap_final.jpg'.format(request.COOKIES['csrftoken']),
        "rectificar":plot_retificar(request,vbs_largas,df,request.POST.get("target") if request.method == 'POST' else request.GET.get("target"), "rectificar_final"),
        "variables":cols
    }

def step_6(request, df = None):
    
    if df is None:
        df = load_pickle(request, 5)
    
    cols = df.columns.tolist()

    for v in request.POST.getlist('removevar'):
        cols.remove(v)
    
    cols.remove(request.POST.get('target'))

    y = df[request.POST.get('target')]
    X = df[cols] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    numeric_cols = X_train.select_dtypes(include=['float64', 'int', 'int64']).columns.to_list()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.to_list()

    # Transformaciones para las variables numéricas
    numeric_transformer = Pipeline(
                        steps=[('scaler', StandardScaler())]
                      )

    categorical_transformer = Pipeline(
                            steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]
                          )

    preprocessor = ColumnTransformer(
                    transformers=[
                        ('numeric', numeric_transformer, numeric_cols),
                        ('cat', categorical_transformer, cat_cols)
                    ],
                    remainder='passthrough'
                )

    # k optimo
    k_range = range(1, 20)
    for k in k_range:
        knn = Pipeline([('preprocessing', preprocessor),
                    ('modelo', KNeighborsClassifier(n_neighbors=k))])
        knn.fit(X_train, y_train)

    param_grid  = {'modelo__n_neighbors': np.linspace(1, 100, 500, dtype=int)}
    knn = Pipeline([('preprocessing', preprocessor),
                 ('modelo', KNeighborsClassifier(n_neighbors=k))])

    grid = RandomizedSearchCV(
        estimator  = knn,
        param_distributions = param_grid 
    )

    grid.fit(X_train, y_train)

    modelo_final = grid.best_estimator_
    y_pred = grid.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred)

    fig = plt.figure()
    plot_confusion_matrix(grid, X_test, y_test)
    plt.savefig('media/{}_knn_matrix.jpg'.format(request.COOKIES['csrftoken']))
    

    #Regresion logística
    C_param_range = [0.001,0.01,0.1]
    for i in C_param_range:
        clf = Pipeline([('preprocessing', preprocessor), 
                    ('modelo', LogisticRegression( max_iter=1000, C = i,random_state = 0))]) 
        clf.fit(X_train, y_train)
         
    clf_predicted = clf.predict(X_test)

    fig = plt.figure()
    plot_confusion_matrix(clf, X_test, y_test)
    plt.savefig('media/{}_clf_matrix.jpg'.format(request.COOKIES['csrftoken']))

    accuracy_rl = accuracy_score(y_test, clf_predicted)

    #Arboles de clasificación
    max_depth_range = list(range(1, 6))
    accuracy = []
    for depth in max_depth_range:   
        clf_arboles = Pipeline([('preprocessing', preprocessor),
                    ('modelo', DecisionTreeClassifier(max_depth = depth, 
                                random_state = 0))])
        clf_arboles.fit(X_train, y_train)    

    # Matrix de confusion 
    clf_arboles_predicted = clf_arboles.predict(X_test)

    fig = plt.figure()
    plot_confusion_matrix(clf_arboles, X_test, y_test)
    plt.savefig('media/{}_tree_matrix.jpg'.format(request.COOKIES['csrftoken']))

    accuracy_arboles = accuracy_score(y_test, clf_arboles_predicted)

    df_transformado = transformacion_data(df[cols],request.POST.get('transformation'))
    df_transformado.to_pickle('media/{}_step6.pkl'.format(request.COOKIES['csrftoken']))

    describe = df_transformado.describe(include="all")
    describe = describe.to_html(classes=classes, justify='center',border=0, )

    info = []
    for c in cols:
        info.append(
            {
                "column":c, 
                "null":df_transformado[c].isnull().sum(), 
                "notnull":df_transformado[c].notnull().sum(), 
                "dtype":str(df_transformado.dtypes[c])
            }
        )

    plt.figure(figsize=(15,10))
    sb.boxplot(data=pd.DataFrame(df_transformado,columns=df_transformado.columns))
    plt.xticks(rotation=90)
    plt.savefig('media/{}_boxplot_trasnformado.jpg'.format(request.COOKIES['csrftoken']))

    vbs_categoricas = df_transformado.columns.tolist()
    vbs_cortas =[]
    vbs_largas =[]

    for v in vbs_categoricas:
        if len(df_transformado[v].unique())<=3:
            vbs_cortas.append(v)
        else:
            vbs_largas.append(v)

    vbs_cortas = np.unique(vbs_cortas).tolist()
    vbs_largas = np.unique(vbs_largas).tolist()

    fig = plt.figure(figsize=(35,35))

    for idx, col in enumerate(vbs_cortas,1):
        ax = fig.add_subplot(plt.subplot(math.ceil(len(vbs_cortas)/4),math.ceil(len(vbs_cortas)/4),idx))
        df_pie = df_transformado.groupby([col]).size().reset_index(name="Count")
        plt.pie(df_pie['Count'], labels=df_pie[col], autopct='%1.1f%%')
        plt.title(col)

    plt.savefig('media/{}_pie_transformado.jpg'.format(request.COOKIES['csrftoken']))

    if len(vbs_cortas) == 1:
        plt.figure(figsize=(5,5))
        sb.boxplot(data=pd.DataFrame(vbs_cortas[0],label='Count'))
        plt.savefig('media/{}_vbs_cortas_transformado.jpg'.format(request.COOKIES['csrftoken']))
    else:
        plt.figure(figsize=(30,30))
        for idx, col in enumerate(vbs_cortas,1):
            plt.subplot(math.ceil(len(vbs_cortas)/4),math.ceil(len(vbs_cortas)/4),idx)
            sb.countplot(x=df_transformado[col],label='Count')
        plt.savefig('media/{}_vbs_cortas_transformado.jpg'.format(request.COOKIES['csrftoken']))


    plt.figure(figsize=(15,10))
    sb.heatmap(df_transformado.corr(),cmap="coolwarm",annot=True)
    plt.savefig('media/{}_heatmap_transformado.jpg'.format(request.COOKIES['csrftoken']))

    accuracy = [accuracy_knn,accuracy_rl,accuracy_arboles]
    
    if accuracy_knn == max(accuracy) :
        best = "KNeighborsClassifier "
    elif accuracy_rl == max(accuracy) :
        best = "LogisticRegression"
    else :
        best = "DecisionTreeClassifier"

    return {
            "describe":describe,
            "target_id": request.POST.get("target_id"),
            "target": request.POST.get("target"),
            "info":info,
            "boxplot":'media/{}_boxplot_trasnformado.jpg'.format(request.COOKIES['csrftoken']),
            "pie":'media/{}_pie_transformado.jpg'.format(request.COOKIES['csrftoken']),
            "vbs_cortas":'media/{}_vbs_cortas_transformado.jpg'.format(request.COOKIES['csrftoken']),
            "heatmap":'media/{}_heatmap_transformado.jpg'.format(request.COOKIES['csrftoken']),
            "rectificar":plot_retificar(request,vbs_largas,df,request.POST.get("target") if request.method == 'POST' else request.GET.get("target"), "rectificar_transformado"),
            "variables":cols,
            "knn":'media/{}_knn_matrix.jpg'.format(request.COOKIES['csrftoken']),
            "clf":'media/{}_clf_matrix.jpg'.format(request.COOKIES['csrftoken']),
            "tree":'media/{}_tree_matrix.jpg'.format(request.COOKIES['csrftoken']),
            "knn_acu": str(round(accuracy_knn*100,2))+"%",
            "clf_acu": str(round(accuracy_rl*100,2))+"%",
            "tree_acu": str(round(accuracy_arboles*100,2))+"%",
            "best":best
        }

def load_pickle(request, step):
    return pd.read_pickle('media/{}_step{}.pkl'.format(request.COOKIES['csrftoken'],step))


def transformacion_data(data,transformar):

    if len(data.select_dtypes(include=['object', 'category']).columns.to_list()) != 0:
        l1 = data.select_dtypes(include=['float64', 'int', 'int64']).columns.to_list()
        l2 = data.select_dtypes(include=['object', 'category']).columns.to_list()
        df1 = data[l1]
        df2 = data[l2]

        for i in df2.columns.tolist():
            le = preprocessing.LabelEncoder()
            le.fit(df2[i])
            df2[i]=le.transform(df2[i])
        
        if transformar == 'StandardScaler':
            data_conct = pd.concat([df1,df2],axis=1)
            scaler = preprocessing.StandardScaler().fit(data_conct)
            data_transformada = pd.DataFrame(scaler.transform(data_conct),columns=data_conct.columns)
            #data_conct = pd.concat([data_transformada,df2],axis=1)
        
        elif transformar == 'MinMaxScaler':
            min_max_scaler = preprocessing.MinMaxScaler()
            data_conct = pd.concat([df1,df2],axis=1)
            data_transformada = pd.DataFrame(min_max_scaler.fit_transform(data_conct),columns=data_conct.columns)
            #data_conct = pd.concat([data_transformada,df2],axis=1)
        
        
        else:
            data_transformada = pd.concat([df1,df2],axis=1)
        
        return(data_transformada)
    
    else:

        if transformar == 'StandardScaler':
            scaler = preprocessing.StandardScaler().fit(data)
            data_transformada = pd.DataFrame(scaler.transform(data),columns=data.columns)
            #data_conct = data_transformada
        
        elif transformar == 'MinMaxScaler':
            min_max_scaler = preprocessing.MinMaxScaler()
            data_transformada = pd.DataFrame(min_max_scaler.fit_transform(data),columns=data.columns)
            #data_conct = data_transformada
        
        else:
            data_transformada = data
        
        return(data_transformada)

def plot_retificar(request,lista,data,clasificar,name):
    try:
        retificar = clasificar in lista
        plt.figure(figsize=(30,30))
        if retificar == False:
            nueva_lista = lista.append(clasificar)
            if len(nueva_lista) <= 3:
                sb.pairplot(data[nueva_lista], hue=clasificar, corner=True ,diag_kind="hist",palette="bright",height=5)
                plt.savefig('media/{}_{}.jpg'.format(request.COOKIES['csrftoken'],name))
            else:
                sb.pairplot(data[nueva_lista], hue=clasificar, corner=True ,diag_kind="hist",palette="bright")
                plt.savefig('media/{}_{}.jpg'.format(request.COOKIES['csrftoken'],name))
        elif retificar == True:
            if len(lista) <= 3:
                sb.pairplot(data[lista], hue=clasificar, corner=True ,diag_kind="hist",palette="bright",height=5)
                plt.savefig('media/{}_{}.jpg'.format(request.COOKIES['csrftoken'],name))
            else:
                sb.pairplot(data[lista], hue=clasificar, corner=True ,diag_kind="hist",palette="bright")
                plt.savefig('media/{}_{}.jpg'.format(request.COOKIES['csrftoken'],name))
        else:
            print('Uppss..')
    except Exception as e:
        return None
        
    return 'media/{}_{}.jpg'.format(request.COOKIES['csrftoken'],name)