import monai
import torch
import tqdm as tqdm 
from monai.transforms import LoadImage, Resize, NormalizeIntensity,Compose,EnsureChannelFirst,AddChannel
from monai.networks.nets import DenseNet121
import argparse
import os
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(device)
model.load_state_dict(torch.load(os.path.join("./model_weight", "best_metric_model.pth")))
model.eval()

class BrainClassificationDatasetinference(torch.utils.data.Dataset):
    def __init__(self, image_files, transforms):
        self.image_files = image_files
        
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index])

    
    
def classificazione(input_image) :
    
    """Summary line.

    La funzione prende in ingresso un file dicom e lo classifica secondo il modello
    Fa solo inferenza ad una singola immagine, non usa il dataloader

    Args:
        input_image: imagine -- passare lista con i vari path delle immagini
     
    Returns: 
    
        ritorna un booleano 
    """
    transforms =Compose([LoadImage(image_only=True),EnsureChannelFirst(),NormalizeIntensity(),Resize((256,256))])
    test_ds = BrainClassificationDatasetinference(input_image,transforms)
    test_loader = DataLoader(test_ds, batch_size=20,num_workers=5)
    with torch.no_grad():
        for test_data in test_loader:
            test_images = (test_data.to(device))
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                if pred[i].item() == 1 : 
                    return True
                else :
                    return False
    
def main () :
    parser=argparse.ArgumentParser(
    description='''Lo script prende in ingresso un file dicom e lo classifica come brain-nobrain. ''')
    parser.add_argument('input_image', help='file dicom da classificare', nargs='+', type=str)
    input_image= parser.parse_args().input_image
    print(classificazione(input_image))
    
    

if __name__ == '__main__' : 
    main()