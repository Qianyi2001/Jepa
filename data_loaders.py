from dataset import *
from sklearn.model_selection import train_test_split

class WallDatasetWithSplits(WallDataset):
    "Assign indices to training dataset and generate splits for encoder training"
    def __init__(self, data_path, split, device="cuda"):
        super().__init__(data_path, device=device)
        num_samples = len(self.states)
        indices = np.arange(num_samples)

        train_idx, test_idx = train_test_split(indices, test_size=0.2) # 80% train, 20% val and test
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5) # 10% val, 10% test

        if split == "train":
            self.indices = train_idx
        elif split == "val":
            self.indices = val_idx
        elif split == "test":
            self.indices = test_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return super().__getitem__(idx)

def create_wall_encoder_dataloader(data_path, split, device="cuda", batch_size=64):
    "For Enconder Only"
    # Split: "train", "val", "test"

    encoder_ds = WallDatasetWithSplits(
        data_path, 
        split, 
        device=device,
    )
    
    encoder_loader = torch.utils.data.DataLoader(
        encoder_ds, 
        batch_size=batch_size, 
        shuffle=(split == "train"), 
        drop_last=True, 
        pin_memory=False,
    )
    
    return encoder_loader

if __name__ == "__main__":

    data_path = "/Users/fionachow/Documents/NYU/CDS/Fall 2024/DS - GA 1008 Deep Learning/Project/DL_Final_Proj/data/subset/"  

    train_loader = create_wall_encoder_dataloader(data_path, split="train", batch_size=64)
    val_loader = create_wall_encoder_dataloader(data_path, split="val", batch_size=64)
    test_loader = create_wall_encoder_dataloader(data_path, split="test", batch_size=64)

    print("Train loader size:", len(train_loader))
    print("Validation loader size:", len(val_loader))
    print("Test loader size:", len(test_loader))
