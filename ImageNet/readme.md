## Data Pre-processing


```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
    './data/ImageNet_path/', split='train', 
        transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=True,
    num_workers=n_worker) # On server 10, num_worker=16 is the fastest

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
    './data/ImageNet_path/', split='val', 
        transforms.Compose([
        transforms.Resize(int(224/0.875)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=n_worker)
```

## Optimizer & Scheduler

```python
# setting 1
epoch = 60
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum= 0.9, weight_decay= 1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

# setting 2
epoch = 60
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum= 0.9, weight_decay= 1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=3, verbose=True, factor=0.3, threshold=1e-4, min_lr=1e-6)
            

```
