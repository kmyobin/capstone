import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from models import BaseModel
import networks
import cv2
import os
import io
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import random

class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
    
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


# 가중치 초기화
def initialize_weights(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        
# Discriminator은 patch gan을 사용합니다.
# Patch Gan: 이미지를 16x16의 패치로 분할하여 각 패치가 진짜인지 가짜인지 식별합니다.
# low-frequency에서 정확도가 향상됩니다.

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage_1 = Dis_block(in_channels*2,64,normalize=False)
        self.stage_2 = Dis_block(64,128)
        self.stage_3 = Dis_block(128,256)
        self.stage_4 = Dis_block(256,512)

        self.patch = nn.Conv2d(512,1,3,padding=1) # 16x16 패치 생성

    def forward(self,a,b):
        x = torch.cat((a,b),1)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.patch(x)
        x = torch.sigmoid(x)
        return x
# UNet
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)),

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x
        
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,4,2,1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat((x,skip),1)
        return x
        
# generator: 가짜 이미지를 생성합니다.
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64,128)                 
        self.down3 = UNetDown(128,256)               
        self.down4 = UNetDown(256,512,dropout=0.5) 
        self.down5 = UNetDown(512,512,dropout=0.5)      
        self.down6 = UNetDown(512,512,dropout=0.5)             
        self.down7 = UNetDown(512,512,dropout=0.5)              
        self.down8 = UNetDown(512,512,normalize=False,dropout=0.5)

        self.up1 = UNetUp(512,512,dropout=0.5)
        self.up2 = UNetUp(1024,512,dropout=0.5)
        self.up3 = UNetUp(1024,512,dropout=0.5)
        self.up4 = UNetUp(1024,512,dropout=0.5)
        self.up5 = UNetUp(1024,256)
        self.up6 = UNetUp(512,128)
        self.up7 = UNetUp(256,64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128,3,4,stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8,d7)
        u2 = self.up2(u1,d6)
        u3 = self.up3(u2,d5)
        u4 = self.up4(u3,d4)
        u5 = self.up5(u4,d3)
        u6 = self.up6(u5,d2)
        u7 = self.up7(u6,d1)
        u8 = self.up8(u7)

        return u8

def preprocess_image(image_path, target_size):
    image = Image.open(image_path)
    image = image.convert("RGB")  # 이미지를 RGB 모드로 변환
    image = image.resize(target_size)  # 이미지를 모델의 입력 크기로 리사이즈
    tensor = F.to_tensor(image)  # 이미지를 텐서로 변환
    tensor = F.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    tensor = tensor.unsqueeze(0)  # 배치 차원 추가
    return tensor
'''
def preprocess_image(image_path, size):
    image = Image.open(image_path)
    # 이미지를 주어진 크기로 변경합니다.
    image = F.resize(image, size)
    # 이미지를 텐서로 변환합니다.
    tensor = F.to_tensor(image)
    # 이미지의 픽셀 값을 -1 ~ 1 범위로 조정합니다.
    tensor = F.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # 텐서를 배치 형태로 변경합니다.
    tensor = tensor.unsqueeze(0)
    return tensor
'''

def visualize_one_sample(test_data, model):
    print("\n\n 5. VISUALIZATION TEST DATA")
    with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
        model.eval() ##### one sample 로 test 할때는 모델 eval 을 넣어줘야함 (아니면 배치 사이즈에 안맞다고 에러뜸)
        r = random.randint(0, len(test_data) - 1)
        X_single_data = torch.FloatTensor(test_data.data[r:r + 1].reshape(-1,3,32,32)).to(device)
        Y_single_data = torch.FloatTensor(test_data.targets[r:r + 1]).to(device)

        print('Label: ', Y_single_data.item())
        single_prediction = model(X_single_data)
        print('Prediction: ', torch.argmax(single_prediction, 1).item())

        plt.imshow(test_data.data[r:r + 1].reshape(32,32,3), interpolation='nearest')
        plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_gen = GeneratorUNet().to(device)
model_dis = Discriminator().to(device)

path2weights_gen = os.path.join('latest_net_G.pth')


# 가중치 불러오기
weights = torch.load(path2weights_gen)
model_gen.load_state_dict(weights)

# evaluation model
model_gen.eval();

#input = cv2.imread('demo_dataset/face.jpg')
#image_tensor = preprocess_image("demo_dataset/face.jpg", (64, 64))

input_image=Image.open('demo_dataset/face.jpg')

# 이미지 전처리
preprocess = transforms.Compose([    
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    transforms.Resize((256, 256)),
])
#input_tensor = preprocess(input_image).unsqueeze(0).to(device)  # 배치 차원 추가
input_tensor=preprocess_image('demo_dataset/face.jpg', (256,256)).to(device)
# 가짜 이미지 생성
with torch.no_grad():
    #input = torch.from_numpy(input).to(device)
    fake_imgs = model_gen(input_tensor).detach().cpu()


plt.imshow(to_pil_image(0.5*fake_imgs[0]+0.5))
plt.axis('off')
plt.show()



# 출력값 후처리를 위한 변환 함수 정의
postprocess = transforms.Compose([
    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    transforms.ToPILImage()
])

# 출력 이미지 후처리
output_image = postprocess(fake_imgs.squeeze(0).cpu())
output_image.save('output.jpg')





'''
# 이미지 변환을 위한 transform 함수 정의
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 모델 생성
state_dict = torch.load('latest_net_G.pth')
model = Pix2PixModel()
model.load_state_dict(state_dict)
model.to(device)

model.eval()

# 이미지 불러오기 및 전처리
input_image = Image.open('demo_dataset/face.jpg').convert('RGB')
input_tensor = transform(input_image).unsqueeze(0).to(device)

# 모델에 입력 이미지 전달하여 출력값 생성
with torch.no_grad():
    output_tensor = model(input_tensor)

# 출력값 후처리를 위한 변환 함수 정의
postprocess = transforms.Compose([
    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    transforms.ToPILImage()
])

# 출력 이미지 후처리
output_image = postprocess(output_tensor.squeeze(0).cpu())

# 결과 이미지 저장
output_image.save('output.jpg')'''