import os, sys
import numpy as np

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


def rmse(y_actual: np.array, y_pred: np.array) -> np.float64:
    """ Root Mean Squared Error of two nympy vectors, or similar.
    """
    return np.sqrt( ((y_actual - y_pred)**2).mean() )

def print_score(model, X_vals, y_vals, sigdig=3):
    """ Print score of trained model
    """
    justify = 8
    # R squared
    r2 = np.round(model.score(X_vals, y_vals), sigdig)
    # RMSE
    y_pred = model.predict(X_vals)
    rmse_pred = np.round(rmse(y_pred, y_vals), sigdig)
    
    print("R^2:".ljust(justify, ' ') + str(r2))
    print("RMSE:".ljust(justify, ' ') + str(rmse_pred))
