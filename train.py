import classifier_utils

def main():
    in_arg = classifier_utils.get_cmd_args()

    dataloaders, class_to_idx = classifier_utils.transform_data(in_arg.data_dir)

    hidden_size = in_arg.hidden_units.split(',')
    hidden_size = [int(x) for x in hidden_size]

    if in_arg.arch == "vgg16":
        arch = {"model":classifier_utils.models.vgg16(pretrained=True),
                "input_size": 25088,
                "name": "vgg16"}
    elif in_arg.arch == "densenet":
        arch = {"model": torch.models.densenet161(pretrained=True),
                "input_size": 2208,
                "name":"densenet"}
    elif in_arg.arch == "alexnet":
        arch = {"model": torch.models.alexnet(pretrained=True),
                "input_size": 9216,
                "name": "alexnet"}
    else: print("model not available!")

    model, criterion, optimizer = classifier_utils.create_model(arch["model"],
                                                class_to_idx,
                                                input_size=arch["input_size"],
                                                hidden_size=hidden_size,
                                                output_size=len(class_to_idx),
                                                lr=in_arg.learning_rate)
    print(model) # debug
    classifier_utils.train_model(model, dataloaders["train"], dataloaders["valid"],
                                optimizer, criterion, epochs = in_arg.epochs,
                                gpu = in_arg.gpu)

    classifier_utils.save_model(arch["name"], model,
                                optimizer, criterion, path=in_arg.save_dir)

if __name__ == "__main__":
    main()
