import monai 

class transform():

    def __init__(self):
        """
        Definition of transformation. 
        We have basic transformation, which we use for training and test/validation, and augmentation, which we use for training only.        
        """
        
        self.__base_transforms = [
                monai.transforms.LoadImaged(keys= ["img"],ensure_channel_first= True,image_only = True), 
                monai.transforms.NormalizeIntensityd(keys = ["img"]) ,
                monai.transforms.SqueezeDimd(keys = ["img"], dim = 3,update_meta=False),
                monai.transforms.ScaleIntensityd(keys = ["img"], minv= 0, maxv= 1) ,
                monai.transforms.EnsureTyped(keys = ["img"]),
                monai.transforms.Resized(keys=["img"], 
                          spatial_size=(256,256),
                           mode = ["area"] )                    
                        
                        ]
        
        self.__augumentation_transform = [
                monai.transforms.RandAxisFlipd(keys = ["img"],
                                            prob = 0.3) , 
                monai.transforms.RandRotated(keys = ["img"], 
                                            prob = 0.3), 
                monai.transforms.Zoomd(keys = ["img"] ,
                                       zoom = 1.3 , 
                                      prob = 0.3)
        ]
    def apply(self, fase: str):
        """
        Applica le trasformazioni. Andare a definire la fase
        Args:
            fase (str): Selezionare : 
                        1. train        --> per la fase di training
                        2. validation   --> per la fase di validazione/test
   
        Returns:
            _type_: _description_
        """
        

        if fase == "train":

                     
            train_transforms = monai.transforms.Compose(self.__base_transforms + self.__augumentation_transform,lazy =False, log_stats=False)

            return train_transforms

        elif fase == "validation":
            return monai.transforms.Compose(self.__base_transforms,lazy =False, log_stats=False)
        
    def get_transformation(self,type : str) :
        
        """
        Get information about transformation

        Args:
            type (str): base or augumentation
        """

        if type == "base" : 
            for i in self.__base_transforms : 
                print(str(i.__class__.__name__))
        elif type == "augumentation" : 
            for i in self.__augumentation_transform : 
                print(str(i.__class__.__name__))
        else : 
            print (f"Chose from :\n 1. base\n 2. augumentation")