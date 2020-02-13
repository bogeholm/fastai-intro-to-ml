import os, sys

def get_basepath(relative_dir_local='data', relative_dir_google='data'):
    """ Return path to base directory depending on whether the
        notebook is running locally, or in Google Colab. If the notebook
        is running in Colab, data is loaded from Google Drive
    """
    GOOGLE_DRIVE_HOME = 'drive/My Drive/' # Equivalent to `cd ~` in Google Drive
    # https://stackoverflow.com/questions/39125532/file-does-not-exist-in-jupyter-notebook
    JUPYTER_HOME =  os.path.abspath('')
    
    if 'google.colab' in sys.modules:
        # Notebook is running in Google Colab
        from google.colab import drive
        drive.mount('/content/drive')
        return os.path.join(GOOGLE_DRIVE_HOME, relative_dir_google)
    else:
        return os.path.join(JUPYTER_HOME, relative_dir_local)