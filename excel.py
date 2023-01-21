import pandas as pd
import os

def concat_dataframe(frame1, frame2):
    if type(frame1) != pd.core.frame.DataFrame:
        return frame2
    elif type(frame2) != pd.core.frame.DataFrame:
        return frame1
    else:
        return pd.concat([frame1, frame2])


def write2excel(OUTPUT_PATH, data, classifier_name, sheet_name='test', per_num=25):
    if os.path.exists(OUTPUT_PATH):
        writer = pd.ExcelWriter(OUTPUT_PATH, mode='a')
    else:
        writer = pd.ExcelWriter(OUTPUT_PATH, mode='a+')
    
    total_data = None
    rounds = 0
    for d in data:
        d_dict = {str(classifier_name[0]+' oa'): d[0][0], str(classifier_name[0]+' aa'): d[0][1], str(classifier_name[0]+' kappa'): d[0][2],
                  str(classifier_name[1]+' oa'): d[1][0], str(classifier_name[1]+' aa'): d[1][1], str(classifier_name[1]+' kappa'): d[1][2]
                  }
        # d_dict = {str(classifier_name[0]+' kappa'): d[0][0],
        #           str(classifier_name[1]+' kappa'): d[1][0]
        #           }
        d_dataframe = pd.DataFrame(d_dict, index=[str(rounds)])
        rounds += per_num
        total_data = concat_dataframe(total_data, d_dataframe)
    
    total_data.to_excel(excel_writer=writer, sheet_name=sheet_name)
    writer.save()
    writer.close()