import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

def get_facenet_model(device='cpu', freeze_features=False):
    """
    Khởi tạo mạng Inception ResNet v1 đã được Pre-train.
    Đầu ra là vector embedding 512 chiều.
    """
    print("⏳ Đang tải mô hình InceptionResnetV1 pre-trained trên VGGFace2...")
    # classify=False nghĩa là ta bỏ lớp Linear cuối cùng, chỉ lấy vector đặc trưng
    model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    
    # Nếu phần cứng yếu, em có thể bật freeze_features=True để đóng băng các lớp đầu
    if freeze_features:
        for name, param in model.named_parameters():
            if 'last_linear' not in name and 'last_bn' not in name:
                param.requires_grad = False
                
    print(f"✅ Đã tải xong mô hình lên thiết bị: {device}")
    return model

def get_triplet_loss(margin=1.0):
    """
    Khởi tạo hàm Triplet Margin Loss.
    Margin là khoảng cách tối thiểu giữa (Anchor, Positive) và (Anchor, Negative).
    """
    # p=2 nghĩa là sử dụng khoảng cách Euclidean (L2 norm)
    return nn.TripletMarginLoss(margin=margin, p=2)