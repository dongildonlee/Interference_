import pandas as pd
import sys
sys.path.append('../')
from packages import objects

def units_for_svm(path, num_units, net, epoch, relu):
     ###########################################################
    ## INPUT:
    #   1. path: path to the csv file
    #   2. num_units: number of units used in SVM
    #   3. net: network ID #
    #   4. epoch: training epoch #
    #   5. relu: relu layer #
    ## OUTPUT:
    #   an array of unit IDs
    ###########################################################
    uoi = pd.read_csv(path+'/'+str(num_units)+' randomly sampled units from distribution of both units in He untrained net'+str(net)+' epoch'+str(epoch)+ ' relu'+str(relu)+'.csv').drop(columns=['Unnamed: 0'])['0'].to_numpy()
    return uoi


def csv_to_obj(csv_filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_filename)

    # Initialize an empty list of units
    units = []

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Initialize a new unit with the id from the current row
        unit = objects.Unit(row['id'])

        # Update the attributes of the unit with the data from the current row
        for attribute in df.columns:
            setattr(unit, attribute, row[attribute])

        # Append the unit to the list of units
        units.append(unit)

    # Return the list of units
    return units



def obj_to_csv(units, filename):
    """
    Converts a list of Unit objects to a DataFrame and saves it to a CSV file.
    
    Parameters:
    - units: A list of Unit objects.
    - filename: The name of the file to save the DataFrame to.
    """

    # Create a dictionary where the keys are the attribute names
    # and the values are lists of the values of those attributes for each Unit
    data = {}
    for attr in vars(units[0]):
        data[attr] = [getattr(unit, attr) for unit in units]
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
