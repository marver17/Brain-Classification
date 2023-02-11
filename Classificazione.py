import monai
import torch
import tqdm as tqdm 
from monai.transforms import ToTensor, Resize, NormalizeIntensity,Compose,EnsureChannelFirst
from monai.networks.nets import DenseNet121
import argparse

class BrainClassificationDatasetinference(torch.utils.data.Dataset):
    def __init__(self, image_files, transforms):
        self.image_files = image_files
        
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index])

    
    
def classificazione(model,trasformazioni,input_image) :
    
    """Summary line.

    La funzione prende in ingresso un file dicom e lo classifica secondo il modello
    Fa solo inferenza ad una singola immagine, non usa il dataloader

    Args:
        model = modello 
        trasformazioni = trasformazioni da usare
        input_image: imagine
     
    Returns: 
    
        ritorna un booleano 
    """
    test_images = trasformazioni(input_image)
    with torch.no_grad():
        test_images.to(device)
        pred = model(test_images).argmax(dim=1).item()
    
    if pred == 1 :
        return True
    else : 
        return False 


    
def main () :
    
    parser=argparse.ArgumentParser(
    description='''Lo script prende in ingresso un file dicom e lo classifica come brain-nobrain. ''')
    parser.add_argument('input_image', help='file dicom da classificare')
    input_image=parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load(os.path.join("./modello", "best_metric_model.pth")))
    model.eval()
    transforms = Compose([LoadImage(input_image),EnsureChannelFirst(),NormalizeIntensity(),Resize((256,256))])
    print(classificazione(model,transforms,input_image))

    

if __name__ == '__main__' : 
    main()