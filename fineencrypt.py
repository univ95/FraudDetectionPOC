import os # a library to manage copying and pasting
import pikepdf # A well written library to work with python
from pikepdf import Pdf

password = 'ConflictRep112423#'


path = 'D:\ProIndrajit\Arrow\Arrow CP - CONFLICTS\11-24-2023'

def protect(file, password=password):
    '''
    This function takes a file as an input and creates another file in the same destination
    This file is encrypted with the password defined earlier
    '''
    pdf = Pdf.open(file)    
    pdf.save(os.path.splitext(file)[0] + '_encrypted.pdf', 
             encryption=pikepdf.Encryption(owner=password, user=password, R=4)) 
    # you can change the 4 to 6 for 256 aes encryption but that file won't open on Acrobat versions lower than 10.0
    pdf.close()
    return
	
	
def remove_originals(file):
    '''
    This will remove the files that don't end with _encrypted.pdf in their names
    '''
    if file.endswith(('.pdf', '.PDF')):
        if not file.endswith('_encrypted.pdf'):
            os.remove(file)



#protecting
for folder, subfolders, files in os.walk(path):
    for file in files:
        if file.endswith(('.pdf', '.PDF')):
            protect(os.path.join(folder, file))
            
#removing originals
for folder, subfolders, files in os.walk(path):
    for file in files:
        if file.endswith(('.pdf', '.PDF')):    
            remove_originals(os.path.join(folder, file))
            
#renaming the encrypted files to match the original filenames
for folder, subfolders, files in os.walk(path):
    for file in files:
        if file.endswith(('.pdf', '.PDF')):
            os.rename(os.path.join(folder, file), os.path.join(folder, file.replace('_encrypted', '')))
