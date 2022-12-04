import torch.nn as nn
import torch.nn.functional as F
import torch


class SN_Net(nn.Module):
    def __init__(self, input_shape):
        super(SN_Net, self).__init__()
        in_ch1 = 1
        out_ch1 = 6
        ker1 = 4
        stride1 = 1
        pad1 = 2

        out_ch2 = 16
        ker2 = 3
        stride2 = 1
        pad2 = 0

        out_ch3 = 32
        ker3 = 2
        stride3 = 1
        pad3 = 0

        out_ch4 = 64
        ker4 = 2
        stride4 = 1
        pad4 = 0

        out_ch5 = 128
        ker5 = 2
        stride5 = 1
        pad5 = 0

        pool_size1 = 3
        pool_size2 = 2
        pool_size3 = 2
        pool_size4 = 2
        pool_size5 = 2

        out_feat1 = 120
        out_feat2 = 84
        out_feat3 = 1

        input_height, input_width = input_shape

        self.conv1 = nn.Conv2d(in_channels = in_ch1, out_channels = out_ch1, kernel_size = ker1, stride = stride1, padding = pad1)
        self.pool1 = nn.MaxPool2d(pool_size1, pool_size1)

        output1_height, output1_width = (input_height - ker1 + 2 * pad1) / stride1 + 1, (input_width - ker1 + 2 * pad1) / stride1 + 1
        output1_height, output1_width = int(output1_height / pool_size1), int(output1_width / pool_size1)

        self.conv2 = nn.Conv2d(in_channels = out_ch1, out_channels = out_ch2, kernel_size = ker2, stride = stride2, padding = pad2)
        self.pool2 = nn.MaxPool2d(pool_size2, pool_size2)

        output2_height, output2_width = (output1_height - ker2 + 2 * pad2) / stride2 + 1, (output1_width - ker2 + 2 * pad2) / stride2 + 1
        output2_height, output2_width = int(output2_height / pool_size2), int(output2_width / pool_size2)

        self.conv3 = nn.Conv2d(in_channels = out_ch2, out_channels = out_ch3, kernel_size = ker3, stride = stride3, padding = pad3)
        self.pool3 = nn.MaxPool2d(pool_size3, pool_size3)

        output3_height, output3_width = (output2_height - ker3 + 2 * pad3) / stride3 + 1, (output2_width - ker3 + 2 * pad3) / stride3 + 1
        output3_height, output3_width = int(output3_height / pool_size3), int(output3_width / pool_size3)

        self.conv4 = nn.Conv2d(in_channels = out_ch3, out_channels = out_ch4, kernel_size = ker4, stride = stride4, padding = pad4)
        self.pool4 = nn.MaxPool2d(pool_size4, pool_size4)

        output4_height, output4_width = (output3_height - ker4 + 2 * pad4) / stride4 + 1, (output3_width - ker4 + 2 * pad4) / stride4 + 1
        output4_height, output4_width = int(output4_height / pool_size4), int(output4_width / pool_size4)

        self.fc1 = nn.Linear(out_ch4 * output4_height * output4_width, out_feat1)
        self.fc2 = nn.Linear(out_feat1, out_feat2)
        self.fc3 = nn.Linear(out_feat2, out_feat3)

    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        #print('1')
        x = self.pool2(F.relu(self.conv2(x)))
        #print('2')
        x = self.pool3(F.relu(self.conv3(x)))
        #print('3')
        x = self.pool4(F.relu(self.conv4(x)))
        #print('4')
        #x = self.pool5(F.relu(self.conv5(x)))
        #print('5')
        x = torch.flatten(x, 1)

        # 이 부분은 변경하셔도 괜찮아요. relu로 할지 sigmoid로 할지
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x