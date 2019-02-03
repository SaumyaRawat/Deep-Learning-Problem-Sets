import torch
import torchvision.transforms as transforms

imgs = np.load("flower_imgs.npy")
labels = LongTensor(np.load("flower_labels.npy")) 
np.random.seed(1234)
np.random.shuffle(arr)
split = int(0.85 * len(arr))

trainX, trainY = imgs[arr[:split]], labels[arr[:split]]
testX, testY = imgs[arr[split:]], labels[arr[split:]]
img_mean = np.mean(np.swapaxes(imgs/255.0,0,1).reshape(3, -1), 1)
img_std = np.std(np.swapaxes(imgs/255.0,0,1).reshape(3, -1), 1)
print("mean: {}, std: {}".format(img_mean, img_std))

class FlowerLoader(torch.utils.data.Dataset):
    def __init__(self, x_arr, y_arr, transform=None):
        self.x_arr = x_arr
        self.y_arr = y_arr
        self.transform = transform

    def __len__(self):
        return self.x_arr.shape[0]

    def __getitem__(self, index):
        img = self.x_arr[index]
        label = self.y_arr[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, label

normalize = transforms.Normalize(mean=list(img_mean),
                                 std=list(img_std))

train_loader = torch.utils.data.DataLoader(
    FlowerLoader(trainX, trainY, transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=True)
# remove augmentation transforms in test loader
test_loader = torch.utils.data.DataLoader(
    FlowerLoader(testX, testY, transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False)

# sample for iterating over loader
for img, label in train_loader:
    img, label = img.to(device), label.to(device)
    pred = net(img)