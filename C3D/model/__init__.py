from utils import *

class C3DNet(nn.Module):
    def __init__(self, num_classes):
        super(C3DNet, self).__init__()
        # output
        self.conv1a = nn.Conv3d(in_channels = 3, out_channels = 64, kernel_size = (3,3,3), stride = 1, padding = 1)
        self.pool1 = nn.MaxPool3d(kernel_size = (1,2,2), stride = (1,2,2)) # exception (1 to 2)

        self.conv2a = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = (3,3,3), stride = 1, padding = 1)
        self.pool2 = nn.MaxPool3d(kernel_size = (2,2,2), stride = (2,2,2))

        self.conv3a = nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = (3,3,3), stride = 1, padding = 1)
        self.conv3b = nn.Conv3d(in_channels = 256, out_channels = 256, kernel_size = (3,3,3), stride = 1, padding = 1)
        self.pool3 = nn.MaxPool3d(kernel_size = (2,2,2), stride = (2,2,2))

        self.conv4a = nn.Conv3d(in_channels = 256, out_channels = 512, kernel_size = (3,3,3), stride = 1, padding = 1)
        self.conv4b = nn.Conv3d(in_channels = 512, out_channels = 512, kernel_size = (3,3,3), stride = 1, padding = 1)
        self.pool4 = nn.MaxPool3d(kernel_size = (2,2,2), stride = (2,2,2))

        self.conv5a = nn.Conv3d(in_channels = 512, out_channels = 512, kernel_size = (3,3,3), stride = 1, padding = 1)
        self.conv5b = nn.Conv3d(in_channels = 512, out_channels = 512, kernel_size = (3,3,3), stride = 1, padding = 1)
        # self.pool5 = nn.MaxPool3d(kernel_size = (2,2,2), stride = (2,2,2), padding = (0,1,1)) # why...?
        self.pool5 = nn.AdaptiveMaxPool3d(output_size = (1,4,4))
        
        self.fc6 = nn.Linear(in_features = 8192, out_features = 4096)
        self.fc7 = nn.Linear(in_features = 4096, out_features = 4096)
        self.fc8 = nn.Linear(in_features = 4096, out_features = num_classes)


        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim = 1)

        self.__init_weight()

    def forward(self, x):
    
        out = self.pool1(self.relu(self.conv1a(x)))
        out = self.pool2(self.relu(self.conv2a(out)))
        out = self.pool3(self.relu(self.conv3b(self.relu(self.conv3a(out)))))
        out = self.pool4(self.relu(self.conv4b(self.relu(self.conv4a(out)))))
        out = self.pool5(self.relu(self.conv5b(self.relu(self.conv5a(out)))))

        out = out.view(-1,8192)
        out = self.dropout(self.relu(self.fc7(self.dropout(self.relu(self.fc6(out))))))
        out = self.fc8(out)
        out = self.softmax(out)

        return out
    
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

def get_1x_lr_params(model):
    b = [model.conv1a, model.conv2a, model.conv3a, model.conv3b, model.conv4a, model.conv4b, model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for j in b[i].parameters():
            if j.requires_grad:
                yield j

def get_10x_lr_params(model):
    b = [model.fc8]
    for i in range(len(b)):
        for j in b[i].parameters():
            if j.requires_grad:
                yield j