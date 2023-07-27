import torch


def get_model(num_output_features: int) -> type[torch.nn.Module]:
    model = torch.hub.load(
        "pytorch/vision", "mobilenet_v2", weights="MobileNet_V2_Weights.DEFAULT"
    )
    final_layer_features = model.classifier[-1].in_features

    # altering the final layer to provide a 2D output
    model.classifier[-1] = torch.nn.Linear(final_layer_features, num_output_features)
    return model
