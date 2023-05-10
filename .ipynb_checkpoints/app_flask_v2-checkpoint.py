from monai.networks.nets import DenseNet121
import torch
import os
from torch.utils.data import DataLoader
from flask import Flask, jsonify, request
import json
from monai.transforms import LoadImage, Resize, NormalizeIntensity,RandRotate,RandFlip,RandZoom,Compose,Activations, AsDiscrete,EnsureChannelFirst,ToTensor,AddChannel
from pydicom.filebase import DicomBytesIO
from pydicom import dcmread
app = Flask(__name__)

transforms = Compose([LoadImage(image_only=True,reader="PydicomReader"),EnsureChannelFirst() ,NormalizeIntensity(),Resize((256,256))])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(device)
root_dir = "./model_weight/"
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
transforms = Compose([AddChannel(),NormalizeIntensity(),Resize((256,256))])
class BrainClassificationDatasetinference(torch.utils.data.Dataset):
    def __init__(self, image_files, transforms):
        self.image_files = image_files
        
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index])

    
def image_conversion(file):
    raw = DicomBytesIO(file.read())
    ds = dcmread(raw)

    return ds.pixel_array



    

def get_prediction(test_images):
    test_images = transforms(test_images)
    with torch.no_grad():
        test_images = test_images.unsqueeze(0).to(device)
        pred = model(test_images).argmax(dim=1)
    return pred.item()    


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = image_conversion(request.files['file'])
        
        # convert that to bytes
        class_name = get_prediction(file)
        return jsonify({ 'class_name': class_name})

    

if __name__ == '__main__':
    app.run()
