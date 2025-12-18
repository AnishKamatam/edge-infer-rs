import torch
from torchvision import models

def main():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        "model/mobilenet_v2.onnx",
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
    )

    print("✅ Exported to model/mobilenet_v2.onnx")

if __name__ == "__main__":
    main()
