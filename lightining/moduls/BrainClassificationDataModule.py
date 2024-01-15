import lightining as L
import os 


class BrainClassificationDataModule(L.DataModule) : 
    super.__init__()
     
    def __init__(self, data_dir : str): 
        self.data_dir = data_dir
        
        
def prepare_data(self) : 
    
    #class_names = ["false","true"]   ### where true are  the images of Brain
    image_files  = {
            "brain"     : [os.path.join(self.data_dir,"true",x) for x in os.listdir(os.path.join(self.data_dir,"true")) if x.endswith(".dcm")] , 
            "no-brain"  : [os.path.join(self.data_dir,"false",x) for x in os.listdir(os.path.join(self.data_dir,"false")) if x.endswith(".dcm")]
    }       