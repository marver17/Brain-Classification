from LightningModule.transformation import transform
import lightning as L
import os 
from sklearn.model_selection import train_test_split
import pandas as pd
from monai.data import CacheDataset, DataLoader
import matplotlib.pylab as plt
import monai

class BrainClassificationDataModule(L.LightningDataModule) : 
     
    def __init__(self, data_dir : str):
        super().__init__() 
        self.data_dir = data_dir
        
        
    def prepare_data(self) : 
        
        image_files  = {
                "brain"     : [os.path.join(self.data_dir,"true",x) for x in os.listdir(os.path.join(self.data_dir,"true")) if x.endswith(".dcm")] , 
                "no-brain"  : [os.path.join(self.data_dir,"false",x) for x in os.listdir(os.path.join(self.data_dir,"false")) if x.endswith(".dcm")]
        }       

        data = {"img": [], "label": []}

        for file_path in image_files["brain"]:
            data["img"].append(file_path)
            data["label"].append(1)

        for file_path in image_files["no-brain"]:
            data["img"].append(file_path)
            data["label"].append(0)

        self.__data = pd.DataFrame(data)
        self.__test_data =  pd.DataFrame([os.path.join(self.data_dir,"test",x) for x in os.listdir(os.path.join(self.data_dir,"test")) if x.endswith(".dcm")], 
                                  columns = ["img"] )
        
        

    def setup(self,stage : str)  : 
        transformation = transform()
        train,validation = train_test_split(self.__data, test_size=0.2, random_state= 42, stratify=self.__data["label"],shuffle=True)
            
        if stage == "fit" : 
            
            self.__train_ds = CacheDataset(train.to_dict("records"),transform=transformation.apply("train"))
            self.__val_ds  = CacheDataset(validation.to_dict("records"), transform= transformation.apply("validation"))

            
        if stage == "validate" : 
            self.__val_ds  = CacheDataset(validation.to_dict("records"), transform= transformation.apply("validation"))

        if stage == "predict" : 
            
            dataset = pd.concat([self.__data["img"]], self.__test_data["img"])
            self.__pred_ds  = CacheDataset(dataset.to_dict("records"), transform= transformation.apply("validation"))
    
    def train_dataloader(self) : 
       return DataLoader(self.__train_ds, batch_size=64,num_workers=20) 
    
    def val_dataloader(self):
        return DataLoader(self.__val_ds, batch_size=64,num_workers=20)

    def predict_dataloader(self):
        return DataLoader(self.__pred_ds, batch_size=64)
    
    def get_label(self) : 
        self.prepare_data()
        train, _ = train_test_split(self.__data, test_size=0.2, random_state= 42, stratify=self.__data["label"],shuffle=True)
        return train["label"].to_list()
    
    def check_dataloader(self) :
        """
        Per eseguire un test del dataloader
        """
        self.prepare_data()
        transformation = transform()
        check_ds = CacheDataset(self.__data.head(1).to_dict("records"),transform=transformation.apply("validation"))

        check_loader = DataLoader(check_ds, batch_size=1, num_workers=1)
        check_data = monai.utils.misc.first(check_loader)
        print(check_data["img"].shape, check_data["label"])
        plt.figure(figsize=(10,10))
        plt.imshow(check_data["img"][0,0,:,:],cmap="gray")
        plt.title("Prova Dataloader")
