# import os
# import io
# import base64

# import torch

# from PIL import Image
# from torchvision import transforms


# class PytorchModel(object):

#     def __init__(self, normalize=False):
#         self.loaded = False
#         self.model = None
#         self.normalize = False

#     def load(self):

#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         MODELNAME = os.environ.get('MODELNAME')
#         VERSION = os.environ.get('VERSION').replace(".","-")
#         MODEL_FILE_NAME = os.environ.get('MODEL_FILE_NAME')

#         MODEL_PATH = f"/app/deployment/{MODELNAME}/{VERSION}-{MODEL_FILE_NAME}"

#         self.model = torch.load(MODEL_PATH)
#         self.model.eval()
#         self.model.to(device)

#         print("Loaded model")

#     def predict(self, X):

#         data = transforms.ToTensor()(Image.open(io.BytesIO(X)))
#         return self._model(data[None, ...]).detach().numpy()

#         decoded = base64.b64decode(X)
#         image = Image.open(io.BytesIO(decoded)).convert("L") 
#         transform = transforms.ToTensor()
#         if self.normalize:
#             transform = transforms.Compose([
#                 transform,
#                 transforms.Normalize((0.1307,), (0.3081,))
#             ])
#         image = transform(image).unsqueeze(0)
#         with torch.no_grad():
#             output = self.model(image)
#         return str(output)


# # class DeepMnist(object):

# #     def __init__(self):
# #         self._model = Net()
# #         self._model.load_state_dict(
# #             torch.load("/storage/model.pt", map_location=torch.device("cpu"))
# #         )
# #         self._model.eval()

# #     def predict(self, X, features_names):
# #         data = transforms.ToTensor()(Image.open(io.BytesIO(X)))
# #         return self._model(data[None, ...]).detach().numpy()

# # import os
# # import io
# # import base64

# # import torch

# # from PIL import Image
# # from torchvision import transforms


# # class PytorchModel(object):

# #     def __init__(self, normalize=False):
# #         self.loaded = False
# #         self.model = None
# #         self.normalize = False


# #     def load(self):

# #         device = "cuda" if torch.cuda.is_available() else "cpu"
# #         MODELNAME = os.environ.get('MODELNAME')
# #         VERSION = os.environ.get('VERSION').replace(".","-")
# #         MODEL_FILE_NAME = os.environ.get('MODEL_FILE_NAME')

# #         MODEL_PATH = f"/app/deployment/{MODELNAME}/{VERSION}-{MODEL_FILE_NAME}"

# #         self.model = torch.load(MODEL_PATH)
# #         self.model.eval()
# #         self.model.to(device)
# #         self.loaded = True
# #         print("Loaded model")

# #     def predicts(self, X):

# #         data = transforms.ToTensor()(Image.open(io.BytesIO(X)))
# #         return self.model(data[None, ...]).detach().numpy()


# import base64
# import io
# import os
# import torch
# from PIL import Image
# import numpy as np
# from torchvision import transforms

# logging.basicConfig(
#     format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)
# logger = logging.getLogger(__name__)


# def load_and_prepare_image(X):
#     img = (Image.open(io.BytesIO(X))).convert("L")
#     img = img.resize((28, 28))
#     img_array = torch.from_numpy(np.array(img))
#     img_array = img_array.float() / 255.0
#     img_array = img_array.unsqueeze(0).unsqueeze(0)
#     return img_array


# class PytorchModel(object):
#     def __init__(self, model):

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.transform = transforms.Compose([
#             transforms.Resize((28, 28)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
    
#     def load(self):
# # specify the path to your .pt file
#         MODELNAME = os.environ.get('MODELNAME')
#         VERSION = os.environ.get('VERSION').replace(".", "-")
#         MODEL_FILE_NAME = os.environ.get('MODEL_FILE_NAME')

#         MODEL_PATH = f"/app/deployment/{MODELNAME}/{VERSION}-{MODEL_FILE_NAME}"
#         self.model = torch.load(MODEL_PATH)
#         # self.model.to(self.device)
#         # self.model.eval()
#         self.loaded = True
#         print("Loaded model")

#     def predict(self, X, feature_names):
#         with torch.no_grad():
#             image_array = load_and_prepare_image(X)
#             prediction = self.model(image_array)
#             _, predicted_class = torch.max(prediction.data, 1)

#         return str(predicted_class.item())
        

# import torch
# from torchvision import transforms
# from PIL import Image
# import logging

# logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)
# logger = logging.getLogger(__name__)

# def load_and_prepare_image(image_path):
#     img = Image.open(image_path).convert("L")  
#     img = img.resize((28, 28))  
#     img_array = torch.from_numpy(np.array(img))
#     img_array = img_array.float() / 255.0  
#     img_array = img_array.unsqueeze(0).unsqueeze(0)  
#     return img_array

# MODEL = torch.load("mnist_cnn_model.pth")

# class DeepMnist(object):

#     def __init__(self):
#         pass

#     def predict(self, *args):
#         image_path = "mnist_sample.1-300x275.jpg" 
        
#         image_array = load_and_prepare_image(image_path)
#         prediction = MODEL(image_array)
#         _, predicted_class = torch.max(prediction.data, 1)
        
#         return str(predicted_class.item())
import os
import io
import torch
from torchvision.transforms import transforms

from PIL import Image

from model import ConvNet

class PytorchModel(object):

    def __init__(self):
        self.device = torch.device('cpu')
        print(f"Loading model for device {self.device}")
        self.loaded = False

    def load(self):

        # MODELNAME = os.environ.get('MODELNAME')
        # VERSION = os.environ.get('VERSION').replace(".", "-")
        # MODEL_FILE_NAME = os.environ.get('MODEL_FILE_NAME')
        self.model = ConvNet()
        # MODEL_PATH = f"./deployment/{MODELNAME}/{VERSION}-{MODEL_FILE_NAME}"
        MODEL_PATH = f"./deployment/demo-model/model_11.pt"
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model = self.model.eval()
        self.model = self.model.to(self.device)
        self.loaded = True
        print("model loaded sucessfully")


    def predict(self, X, *args):

        if not self.loaded:
            self.load()
        with torch.no_grad():
            data = transforms.ToTensor()(Image.open(io.BytesIO(X)).convert('L'))
            output = self.model(data)
            pred = output.argmax(dim=1, keepdim=True)
            return pred.item()
