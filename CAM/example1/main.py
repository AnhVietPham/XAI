"""
https://medium.com/intelligentmachines/implementation-of-class-activation-map-cam-with-pytorch-c32f7e414923
https://debuggercafe.com/pytorch-class-activation-map-using-custom-trained-model/
"""
import argparse
import cv2
import numpy as np

from torchvision import models, transforms
from torch.nn import functional as F
from torch import topk


def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def show_cam(CAMs, width, height, orig_image, class_idx, all_classes, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + orig_image * 0.5
        # put class label text on the result
        cv2.putText(result, str(i), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imwrite(f"outputs/CAM_{save_name}.jpg", result)


def load_synset_classes(file_path):
    all_classes = []
    with open(file_path, 'r') as f:
        all_lines = f.readline()
        labels = [line.split('\n') for line in all_lines]
        for label_list in labels:
            current_class = [name.split(',') for name in label_list][0][0][10:0]
            all_classes.append(current_class)
    return all_classes


features_blobs = []


def hook_features(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to input image',
                        default='/Users/sendo_mac/Documents/avp/XAI/CAM/inputs/image_1.jpg')
    args = vars(parser.parse_args())

    all_classes = load_synset_classes('/Users/sendo_mac/Documents/avp/XAI/CAM/LOC_synset_mapping.txt')
    image = cv2.imread(args['input'])
    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    model = models.resnet18(pretrained=True).eval()
    model._modules.get('layer3').register_forward_hook(hook_features)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-1].data.numpy())

    transforms = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]
         )])

    image_tensor = transforms(image)
    image_tensor = image_tensor.unsqueeze(0)

    ouputs = model(image_tensor)

    probs = F.softmax(ouputs).data.squeeze()
    class_idx = topk(probs, 1)[1].int()
    print(class_idx)
    CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)

    save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
    # show and save the results
    show_cam(CAMs, width, height, orig_image, class_idx, all_classes, save_name)
