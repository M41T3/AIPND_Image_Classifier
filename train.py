import classifier_utils

def main():
    in_arg = classifier_utils.get_cmd_args()
    print(in_arg.data_dir)

    dataloaders, class_to_idx = classifier_utils.transform_data(in_arg.data_dir)

    hidden_size = in_arg.hidden_units.split(',')
    hidden_size = [int(x) for x in hidden_size]

    models = {"vgg16":     {"model":torch.models.vgg16(pretrained=True),
                            "input_size": 25088},
              "densenet":  {"model": torch.models.densenet161(pretrained=True),
                            "input_size": 2208},
              "alexnet":    {"model": torch.models.alexnet(pretrained=True),
                            "input_size": 9216}}

    model, criterion, optimizer = classifier_utils.create_model(models[in_arg.arch]["model"],
                                                class_to_idx,
                                                input_size=models[in_arg.arch]["input_size"],
                                                hidden_size=hidden_size,
                                                output_size=len(class_to_idx),
                                                lr=in_arg.learning_rate)

    classifier_utils.train_model(model, dataloader["train"], dataloader["valid"],
                                optimizer, criterion, epochs = in_arg.epochs,
                                gpu = in_arg.gpu)

    classifier_utils.save_model(model_name, model, optimizer, criterion, path=in_arg.save_dir)

if __name__ == "__main__":
    main()
