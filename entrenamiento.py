import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import torchvision.models.detection.backbone_utils as backbone_utils
from torchstat import stat
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import datasets, models, transforms
import torch.nn as nn
import pandas as pd
import time
import pynvml
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import seaborn as sns
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import Subset
from random import sample
import torchvision
from gpustat import GPUStatCollection
import psutil


class GalaxyDataset(Dataset):
    def __init__(self, csv_file, root, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.root = root
        self.transforms = transforms
        self.class_mapping = {'E': 1, 'S': 2, 'SB': 3, 'M': 4}

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, 'noirImages', self.data.iloc[idx]['image_filename'])
        mask_name = os.path.join(self.root, 'noirMasks', self.data.iloc[idx]['mask_filename'])
        
        img = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name)
        mask = np.array(mask)
        
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # Eliminar el fondo
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        labels = []
        
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            class_label = self.data.iloc[idx]['class']
            labels.append(self.class_mapping[class_label])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area, "iscrowd": iscrowd}
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.data)

def get_gpu_usage():
    gpu_stats = GPUStatCollection.new_query()
    gpu_usage = gpu_stats.jsonify()["gpus"][0]
    return {
        "gpu_utilization": gpu_usage["utilization.gpu"],
        "gpu_memory_used": gpu_usage["memory.used"],
        "gpu_temperature": gpu_usage["temperature.gpu"]
    }

def get_cpu_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    cpu_memory = psutil.virtual_memory().used
    return {
        "cpu_utilization": cpu_usage,
        "cpu_memory_used": cpu_memory
    }


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def prepare_data_loader(root_dir, train_csv='train_noir.csv', val_csv='validation_noir.csv', test_csv='test_noir.csv'):
    # Cargar datasets
    train_dataset = GalaxyDataset(csv_file=os.path.join(root_dir, train_csv), root=root_dir, transforms=get_transform(train=True))
    val_dataset = GalaxyDataset(csv_file=os.path.join(root_dir, val_csv), root=root_dir, transforms=get_transform(train=False))
    test_dataset = GalaxyDataset(csv_file=os.path.join(root_dir, test_csv), root=root_dir, transforms=get_transform(train=False))
    
    # Imprimir la cantidad de datos en cada división
    print(f"Cantidad de datos en el conjunto de entrenamiento: {len(train_dataset)}")
    print(f"Cantidad de datos en el conjunto de validación: {len(val_dataset)}")
    print(f"Cantidad de datos en el conjunto de prueba: {len(test_dataset)}")

    # Crear data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    
    return train_loader, val_loader, test_loader

def load_compressed_model(_path, _model):
    model_State_dict = torch.load(_path + _model)
    model_State_dict.eval()
    return model_State_dict

def create_faster_rcnn_model(backbone, num_classes=5):
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

def save_model(model, _path, _model):
    torch.save(model, _path + _model)

def measure_energy_consumption(model, data_loader, device):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    start_time = time.time()
    
    model.eval()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        with torch.no_grad():
            outputs = model(images)
    
    end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    end_time = time.time()
    
    pynvml.nvmlShutdown()

    total_energy = (end_energy - start_energy) / 1000.0 # Convert to joules
    total_time = end_time - start_time
    co2_emissions = total_energy * 0.000707 # Estimation: 0.000707 kg CO2 per joule (adjust according to your region)

    df_energy = pd.DataFrame({
        "total_energy": [total_energy],
        "total_time": [total_time],
        "co2_emissions": [co2_emissions]
    })
    df_energy.to_csv("metricas/new_tensor_faster_multiclase_resnet_101_20ep_inference_energy.csv", index=False)

    print(f"Total energy consumption (in joules): {total_energy:.4f}")
    print(f"Total CO2 emissions (in kg): {co2_emissions:.4f}")

# Función para entrenar el modelo con un subconjunto de datos
def train_model(model, train_loader, val_loader, device, num_epochs=20, subset_size=None):
    if subset_size:
        train_loader = DataLoader(Subset(train_loader.dataset, range(subset_size)), batch_size=train_loader.batch_size, shuffle=True, collate_fn=train_loader.collate_fn)
        val_loader = DataLoader(Subset(val_loader.dataset, range(subset_size)), batch_size=val_loader.batch_size, shuffle=False, collate_fn=val_loader.collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    metrics = []

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    for epoch in range(num_epochs):
        start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        start_time = time.time()

        train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        _model_save_path = 'modelos/Finales/'
        _model_name = f'test_epoch_{epoch}'
        save_model(model, _model_save_path, _model_name)

        end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        end_time = time.time()

        energy_used = (end_energy - start_energy) / 1000.0  # Convert to joules
        co2_emissions = energy_used * 0.000707  # Estimation: 0.000707 kg CO2 per joule (adjust according to your region)

        # Obtener métricas de uso de GPU y CPU
        gpu_metrics = get_gpu_usage()
        cpu_metrics = get_cpu_usage()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics.meters['loss'].global_avg,
            "energy_used": energy_used,
            "co2_emissions": co2_emissions,
            "gpu_utilization": gpu_metrics["gpu_utilization"],
            "gpu_memory_used": gpu_metrics["gpu_memory_used"],
            "gpu_temperature": gpu_metrics["gpu_temperature"],
            "cpu_utilization": cpu_metrics["cpu_utilization"],
            "cpu_memory_used": cpu_metrics["cpu_memory_used"],
            "time_elapsed": end_time - start_time
        }

        metrics.append(epoch_metrics)

        # Imprimir los resultados de la época en pantalla
        train_loss = epoch_metrics['train_loss'] if epoch_metrics['train_loss'] is not None else 0.0
        energy_used = epoch_metrics['energy_used'] if epoch_metrics['energy_used'] is not None else 0.0
        co2_emissions = epoch_metrics['co2_emissions'] if epoch_metrics['co2_emissions'] is not None else 0.0
        time_elapsed = epoch_metrics['time_elapsed'] if epoch_metrics['time_elapsed'] is not None else 0.0

        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Energy Used (J): {energy_used:.2f}, CO2 Emissions (kg): {co2_emissions:.6f}, Time Elapsed: {time_elapsed:.2f}s")
        print(f"GPU Utilization: {epoch_metrics['gpu_utilization']}%, GPU Memory Used: {epoch_metrics['gpu_memory_used']}MB, GPU Temperature: {epoch_metrics['gpu_temperature']}C")
        print(f"CPU Utilization: {epoch_metrics['cpu_utilization']}%, CPU Memory Used: {epoch_metrics['cpu_memory_used'] / (1024 ** 3):.2f}GB")

    pynvml.nvmlShutdown()

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv("metricas/new_tensor_faster_multiclase_resnet_101_20ep.csv", index=False)

    return model


def main():
    pynvml.nvmlInit()
    #_path = 'resnet/50_20ep/'
    _path = 'tucker/modelo_comprimidos/'
    _model = 'TENSOR_Entire_Model_RESNET_101_NOIR_20_FINAL.pth'
    resnet_net = load_compressed_model(_path, _model)
    modules = list(resnet_net.children())[:-1]
    backbone = nn.Sequential(*modules)
    backbone.out_channels = 2048
    resnet_net_fpn = backbone_utils.resnet_fpn_backbone('resnet50', resnet_net)
    root_dir = 'dataset/'
    num_classes = 5
    model = create_faster_rcnn_model(resnet_net_fpn, num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    train_loader, val_loader, test_loader=prepare_data_loader(root_dir, train_csv='train_noir.csv', val_csv='validation_noir.csv', test_csv='test_noir.csv')
    trained_model = train_model(model, train_loader, val_loader, device, num_epochs=20, subset_size=None)
    _model_save_path = 'modelos/Finales/'
    _model_name = 'new_tensor_faster_multiclase_resnet_101_20ep'
    save_model(trained_model, _model_save_path, _model_name)

if __name__ == "__main__":
    main()
