import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
import os

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
        df.columns = [i.title().strip().replace(' ','_') for i in df.columns.to_list()]

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


    return {
        "info": info,
        "describe":describe,
        "boxplot":'media/{}_boxplot.jpg'.format(request.COOKIES['csrftoken']),
        "vbs_cortas":'media/{}_vbs_cortas.jpg'.format(request.COOKIES['csrftoken'])
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
            pass

    return {
        "target_id": int(request.GET.get('target')),
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
        
        try:
            
            for r in request.POST.keys():
                if r.find("out_") >=0 and request.POST.getlist(r):
                    dic_out.setdefault(r.split("out_")[1],pd.Series(request.POST.getlist(r), dtype=df[r.split("out_")[1]].dtype).tolist())

                if r.find("imp_") >=0 and request.POST.getlist(r):
                    array = np.array(request.POST.get(r))
                    dic_imp.setdefault(r.split("imp_")[1],array)

            df = df[~ df.isin(dic_out)]
            df = df.fillna(dic_imp)
            df = df.dropna()

        except Exception as e:
            print(e)
            pass

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
    plt.savefig('media/{}_heatmap.jpg'.format(request.COOKIES['csrftoken']))

    cols = df.columns.tolist()
    target = cols[int(request.POST.get('target'))]
    cols.remove(target)

    return {
        "describe":describe,
        "target": request.POST.get("target"),
        "info":info,
        "boxplot":'media/{}_boxplot_final.jpg'.format(request.COOKIES['csrftoken']),
        "pie":'media/{}_pie_final.jpg'.format(request.COOKIES['csrftoken']),
        "vbs_cortas":'media/{}_vbs_cortas_final.jpg'.format(request.COOKIES['csrftoken']),
        "heatmap":'media/{}_heatmap.jpg'.format(request.COOKIES['csrftoken']),
        "variables":cols
    }
    

def load_pickle(request, step):
    return pd.read_pickle('media/{}_step{}.pkl'.format(request.COOKIES['csrftoken'],step))