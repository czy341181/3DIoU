from iou3d import iou3d_utils
import torch
import numpy as np
def read_gt3dBox(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    box3d_result = []
    for line in lines:
        line = line.split(' ')
        cls = str(line[0])
        if cls =='DontCare':
            continue
        location_x = torch.FloatTensor([float(line[11])])
        location_y = torch.FloatTensor([float(line[12])])
        location_z = torch.FloatTensor([float(line[13])])
        size_x = torch.FloatTensor([float(line[8])])
        size_y = torch.FloatTensor([float(line[9])])
        size_z = torch.FloatTensor([float(line[10])])
        rotate = torch.FloatTensor([float(line[14])])
        box3d = []
        box3d = [location_x, location_y, location_z, size_x, size_y, size_z, rotate]
        box3d = torch.cat(box3d, dim=-1).reshape(-1, 7)
        box3d_result.append(box3d)
    gt_box = torch.cat(box3d_result, dim=0)
    return gt_box




# if __name__ == "__main__":
#     data_path = "/home/czy/czy/data/KITTI/object/training/label_2/000018.txt"
#     gt_box = read_gt3dBox(data_path)
#     print("gt_box:",gt_box.shape)
#
#     iou_result = iou3d_utils.boxes_iou3d_cpu_test(gt_box,gt_box)
#     #iou_result = iou3d_utils.boxes_iou3d_gpu_test(gt_box.cuda(),gt_box.cuda())
#     #iou_result = iou3d_utils.boxes_aligned_iou3d_gpu(gt_box.cuda(),gt_box.cuda())
#     print(iou_result)

if __name__ == '__main__':
    import numpy as np
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    box_a = torch.Tensor([1, 1, 1, 2, 2, 2, np.pi / 4]).view(-1, 7).float().cuda().repeat(2, 1)
    box_b = torch.Tensor([0, 0, 0, 1, 1, 1, 0]).view(-1, 7).float().cuda().repeat(2, 1)

    boxes_a = torch.rand(20, 7) * torch.tensor([10, 10, 3, 4, 4, 4, np.pi], dtype=torch.float32) \
              + torch.tensor([0, -10, -1, 0, 0, 0, -np.pi], dtype=torch.float32)
    boxes_b = torch.rand(20, 7) * torch.tensor([10, 10, 3, 4, 4, 4, np.pi], dtype=torch.float32) \
              + torch.tensor([0, -10, -1, 0, 0, 0, -np.pi], dtype=torch.float32)

    iou3d_0, iou_bev_0 = iou3d_utils.boxes_iou3d_gpu(boxes_a.cuda(), boxes_b.cuda(), rect=False, need_bev=True)
    print(iou_bev_0, iou3d_0)

    #import ipdb; ipdb.set_trace()

    iou3d = iou3d_utils.boxes_aligned_iou3d_gpu(boxes_a.cuda(), boxes_b.cuda(), rect=False)

    print(iou3d)