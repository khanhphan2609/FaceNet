import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Lấy danh sách tên các thư mục (mỗi thư mục là 1 người)
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_images = {}
        
        # Lưu toàn bộ đường dẫn ảnh theo từng class
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            images = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            # Chỉ lấy những người có từ 2 ảnh trở lên (để có Anchor và Positive)
            if len(images) >= 2:
                self.class_to_images[cls] = images

        # Cập nhật lại danh sách classes hợp lệ
        self.valid_classes = list(self.class_to_images.keys())

        # Định nghĩa phép biến đổi Tensor (Rất quan trọng)
        # 1. Chuyển PIL Image thành Tensor dải [0, 1]
        # 2. Chuẩn hóa về dải [-1, 1] để phù hợp với InceptionResnetV1
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)), # Đảm bảo ảnh luôn là 160x160
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        # Chiều dài của dataset mang tính quy ước. 
        # Thầy lấy tổng số lượng ảnh làm độ dài của 1 epoch.
        return sum([len(imgs) for imgs in self.class_to_images.values()])

    def __getitem__(self, idx):
        # 1. Chọn ngẫu nhiên 1 người làm Anchor
        anchor_cls = random.choice(self.valid_classes)
        
        # 2. Bốc ngẫu nhiên 2 ảnh khác nhau của người đó (Anchor và Positive)
        anchor_img_path, pos_img_path = random.sample(self.class_to_images[anchor_cls], 2)

        # 3. Chọn ngẫu nhiên 1 người khác làm Negative
        neg_cls = random.choice(self.valid_classes)
        while neg_cls == anchor_cls: # Đảm bảo người Negative phải khác người Anchor
            neg_cls = random.choice(self.valid_classes)
            
        # 4. Bốc ngẫu nhiên 1 ảnh của người Negative
        neg_img_path = random.choice(self.class_to_images[neg_cls])

        # Đọc ảnh bằng PIL
        anchor_img = Image.open(anchor_img_path).convert('RGB')
        pos_img = Image.open(pos_img_path).convert('RGB')
        neg_img = Image.open(neg_img_path).convert('RGB')

        # Áp dụng chuẩn hóa Tensor
        anchor_tensor = self.transform(anchor_img)
        pos_tensor = self.transform(pos_img)
        neg_tensor = self.transform(neg_img)

        return anchor_tensor, pos_tensor, neg_tensor