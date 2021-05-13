import pandas as pd
import numpy as np

def handle_uploaded_file(f):  
    with open('media/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():  
            destination.write(chunk)
    return destination.name

def get_df(file):
    classes = 'table table-striped table-hover table-sm text-center'

    df = pd.read_csv(file)
    df.columns = [i.title().strip().replace(' ','_') for i in df.columns.to_list()]

    nulls = ['Null','NaN','',' ','#N/A','#N/D','?','#']
    df = df.replace(nulls,np.nan)

    describe = df.describe()
    describe = describe.to_html(classes=classes, justify='center',border=0, )

    head = df.head()
    head = head.to_html(classes=classes, justify='center',border=0, index=False)
    
    sample = df.sample(10)
    sample = sample.to_html(classes=classes, justify='center',border=0, index=False)

    n_rows = df.shape[0]
    n_cols = df.shape[1]
    cols = df.columns.to_list()

    data_frame = {
        'describe':describe,
        'head':head,
        'sample':sample,
        'n_rows':n_rows,
        'n_cols':n_cols,
        'cols':cols,
    }

    return data_frame