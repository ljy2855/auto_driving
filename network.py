import torch
from torch import nn
import numpy as np

class ClassificationNetworkColors(torch.nn.Module):
    def __init__(self):

        super().__init__()
        # setting device on GPU if available, else CPU
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.classes = [[-1., 0., 0.],  # left
                        [-1., 0.5, 0.], # left and accelerate
                        [-1., 0., 0.8], # left and brake
                        [1., 0., 0.],   # right
                        [1., 0.5, 0.],  # right and accelerate
                        [1., 0., 0.8],  # right and brake
                        [0., 0., 0.],   # no input
                        [0., 0.5, 0.],  # accelerate
                        [0., 0., 0.8]]  # brake
        self.num_actions = len(self.classes)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        feature_size =  4096


        self.fc = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        # """
        # D : Network Implementation

        # Implementation of the network layers. 
        # The image size of the input observations is 96x96 pixels.

        # Using torch.nn.Sequential(), implement each convolution layers and Linear layers
        # """

        # convolution layers 
        # Linear layers (output size : 9)





    def forward(self, observation):
        # """

        # D : Network Implementation

        # The forward pass of the network. 
        # Returns the prediction for the given input observation.
        # observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        # return         torch.Tensor of size (batch_size, C)

        # """
    
        observation = observation.permute(0, 3, 1, 2)
        x = self.conv(observation)  # 컨볼루션 레이어 적용
        # x = x.view(x.size(0), -1)  # 플래트닝
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)  # 전결합 레이어 적용
        return x

    def actions_to_classes(self, actions):
        # """
        # C : Conversion from action to classes

        # For a given set of actions map every action to its corresponding
        # action-class representation. Every action is represented by a 1-dim vector 
        # with the entry corresponding to the class number.
        # actions:        python list of N torch.Tensors of size 3
        # return          python list of N torch.Tensors of size 1
        # """
        class_tensors = []
        for action in actions:
            # action을 numpy 배열로 변환
            action_np = np.array(action)

            # 각 클래스와의 거리 계산
            distances = [np.linalg.norm(action_np - np.array(class_action)) for class_action in self.classes]

            # 가장 거리가 작은 클래스의 인덱스 찾기
            class_index = np.argmin(distances)

            # 클래스 번호를 텐서로 변환하여 리스트에 추가
            class_tensors.append(torch.tensor([class_index]))
        # print(class_tensors)
        return class_tensors




    def scores_to_action(self, scores):
        # """
        # C : Selection of action from scores

        # Maps the scores predicted by the network to an action-class and returns
        # the corresponding action [steer, gas, brake].
        #                 C = number of classes
        # scores:         python list of torch.Tensors of size C
        # return          (float, float, float)
        # """


        scores_flattened = scores.flatten()

        # 가장 높은 점수를 가진 클래스의 인덱스 찾기
        _, predicted_class_index = scores_flattened.max(0)

        # 인덱스를 스칼라 값으로 변환
        predicted_class_index = predicted_class_index.item()





        action = self.classes[predicted_class_index]

        return tuple(action)

    
