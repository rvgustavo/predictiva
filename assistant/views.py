from django.shortcuts import render
from django.shortcuts import redirect

from .forms import data_form
from .functions import handle_uploaded_file, step_1, step_2, step_3, step_4, step_5, step_6, load_pickle

IMAGE_FILE_TYPES = ['csv', 'data'] 

def home(request):
    return render(request,'assistant/home.html')

def step_assistant(request, step=1):
    if request.method == 'POST':
        if(step==1):
            form = data_form()
            form = data_form(request.POST, request.FILES)
            if(len(request.FILES) == 0):
                return render(request, 'assistant/error.html',{'msg': "You must select a file...", "return":"/assistant/1"})
            if form.is_valid():
                file_name = request.FILES['data_file'].name
                file_type = file_name.split('.')[-1]
                file_type = file_type.lower()

                if file_type not in IMAGE_FILE_TYPES:
                    return render(request, 'assistant/error.html',{'msg': "Invalid format file..."})

                path_file = handle_uploaded_file(request.FILES['data_file'])
                df = step_1(request, path_file)            
                return render(request, 'assistant/step1.html', {'uploaded': True, 'df':df, 'file_name':file_name})
        if(step==2):
            return render(request,'assistant/step2.html')
        if(step==3):
            return render(request,'assistant/step3.html')
        if(step==4):
            return render(request,'assistant/step4.html')
        if(step==5):
            data = step_5(request)
            return render(request,'assistant/step5.html',{'data':data})
        if(step==6):
            data = step_6(request)
            return render(request,'assistant/step6.html',{'data':data})
                
    else:  
        if(step==1):
            if request.GET.get('nav'):
                try:
                    df = step_1(request, df=load_pickle(request, step))                
                    return render(request, 'assistant/step1.html', {'uploaded': True, 'df':df})
                except:
                   return render(request, 'assistant/error.html',{'msg': "The session has been expired", "return":"/home"}) 
            else:
                return render(request,'assistant/step1.html')
        if(step==2):
            try:
                df = step_2(request)
                return render(request,'assistant/step2.html',{'df':df})
            except:
                return render(request, 'assistant/error.html',{'msg': "The session has been expired", "return":"/home"}) 
        if(step==3):
            try:
                cols = step_3(request)
                return render(request,'assistant/step3.html',{'cols':cols, "target":request.GET.get("target")})
            except:
                return render(request, 'assistant/error.html',{'msg': "The session has been expired", "return":"/home"}) 
        if(step==4):
            if request.GET.get('target'):
                try:
                    data = step_4(request)
                    return render(request,'assistant/step4.html',{'data':data})
                except Exception as e:
                    return render(request, 'assistant/error.html',{'msg': "The session has been expired", "return":"/home"})
            else:
                return redirect('/assistant/3')
        if(step==5):
            if request.GET.get('nav'):
                try:
                    data = step_5(request, df=load_pickle(request, step))                
                    return render(request,'assistant/step5.html',{'data':data})
                except:
                   return render(request, 'assistant/error.html',{'msg': "The session has been expired", "return":"/home"}) 
            else:
                return redirect('/assistant/4')
        if(step==6):
            return redirect('/assistant/5')
            

