import os
import json

def create_folder(root_dir):
    """
    Creates folder in whiche results will be saved
    in :  directory where to create the folder
    out : / creatin of a folder
    """

    dir_name = root_dir
    print(root_dir)
    try:
        print('here')
        os.mkdir(dir_name)
        print("Directory " , dir_name ,  " Created ")
    except FileExistsError:
        print("Directory " , dir_name ,  " already exists")


def save_logs(input_dict):
    """
    Creates and saves a log file with all information of the run
    in : dico like {module:'python file'
                    input: {date : '', data: 'split' or 'all', loss_type :'',
                    batch_size: '', nb_epoch:'', root_dir:'',
                    input_type:'jeff crops or normalized crops',
                    skeleton: 'True or False'},
                    output: {loss_train:'', loss_test:''}}
    """
    root_dir = input_dict['input']['root_dir']

    log_file = open(root_dir + "logs.json", "a+")
    log_file.write(json.dumps(input_dict))
    log_file.close()
