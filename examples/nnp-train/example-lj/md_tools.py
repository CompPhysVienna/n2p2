import numpy as np

def readColsFromFile(file_name, col_dict):
    """ Read multiple columns of data from file.
    
    Parameters
    ----------
    file_name : string
        Name of the data file.
    col_dict : dict
        Dictionary containing (column_name, column_number) pairs,
        e.g. {"Temp" : 3}. Column numbers start with 1.
        
    Returns
    -------
    dict
        Dictionary with the data for all requested columns,
        e.g. {"Temp" : np.array(120.0, 121.3, ...)}.

    Note: all lines in the data file starting with '#' are ignored.
    """
    data = {col_name : [] for col_name in col_dict}
    f = open(file_name, "r")
    for line in f:
        split_line = line.split()
        if split_line[0][0] == "#":
            continue
        for col_name, col in col_dict.iteritems():
            data[col_name].append(float(split_line[col-1]))
    f.close()
    data = {col_name : np.array(col_data) for col_name, col_data
            in data.iteritems()}
    return data
