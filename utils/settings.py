import os
import base64



# -------------------------------- Gera link para download do dataset ----
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
    """
    csv = df.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    return b64



# ----------------------------- Configurações de cache de leitura de arquivo
class FileReference:

    def __init__(self, filename):
        self.filename = filename


def hash_file_reference(file_reference):
    filename = file_reference.filename
    return (filename, os.path.getmtime(filename))
