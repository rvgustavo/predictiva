from django import forms

#DataFlair #File_Upload
class data_form(forms.Form):
    data_file = forms.FileField()