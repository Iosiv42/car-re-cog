__all__ = [
    "main_cli",
]

import argparse

CLASSES = (
    "matiz black", "matiz blue", "matiz red",
    "rio black", "rio blue", "rio red",
    "tiggo black", "tiggo blue", "tiggo red",
)


def main_cli():
    import json

    import torch
    import torchvision
    import torchvision.transforms as transforms

    from PIL import Image

    parser = argparse.ArgumentParser(
        prog="car-recog",
        description="Recognize car model based on specified"
                    "convolution neural network model",
        epilog="Hope it help helps you"
    )

    parser.add_argument("input_image")
    parser.add_argument("path_to_model")
    parser.add_argument("--normalize_params_path", "-np", required=False)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = torchvision.models.resnet18()
    ftrs_count = net.fc.in_features
    classes_count = 9
    net.fc = torch.nn.Linear(ftrs_count, classes_count)
    net.load_state_dict(torch.load(
        args.path_to_model, weights_only=True
    ))
    net = net.to(device)
    net.eval()

    if args.normalize_params_path is None:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        with open(args.normalize_params_path) as f:
            mean, std = json.load(f)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    img = transform(Image.open(args.input_image).convert("RGB"))

    output = net(img.resize(1, 3, 224, 224).to(device))
    _, predicted = torch.max(output, 1)
    print(CLASSES[predicted[0]])
