from django.shortcuts import render

from .forms import data_form
from .functions import handle_uploaded_file, get_df 

IMAGE_FILE_TYPES = ['csv', 'data', 'jpg', 'jpeg'] 

def home(request):
    form = data_form()
    if request.method == 'POST':
        form = data_form(request.POST, request.FILES)
        if(len(request.FILES) == 0):
            return render(request, 'assistant/error.html',{'msg': "You must select a file..."})
        if form.is_valid():
            file_name = request.FILES['data_file'].name
            file_type = file_name.split('.')[-1]
            file_type = file_type.lower()

            if file_type not in IMAGE_FILE_TYPES:
                 return render(request, 'assistant/error.html',{'msg': "Invalid format file..."})

            path_file = handle_uploaded_file(request.FILES['data_file'])
            df = get_df(path_file)            
            return render(request, 'assistant/home.html', {'uploaded': True, 'df':df, 'file_name':file_name})
    else:  
        return render(request, 'assistant/home.html') 

